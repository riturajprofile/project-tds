# main.py
import os
import re
import json
import base64
import stat
import shutil
import asyncio
import logging
import sys
from typing import List, Optional
from datetime import datetime

import httpx
import git
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Import validation module
try:
    from validation import validate_generated_code, get_validation_feedback
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    logging.warning("Validation module not available. Install beautifulsoup4 for pre-deployment validation.")

# ------------------------- Settings -------------------------
class Settings(BaseSettings):
    GEMINI_API_KEY: str = Field("", env="GEMINI_API_KEY")
    GITHUB_TOKEN: str = Field("", env="GITHUB_TOKEN")
    GITHUB_USERNAME: str = Field("", env="GITHUB_USERNAME")
    STUDENT_SECRET: str = Field("", env="STUDENT_SECRET")
    LOG_FILE_PATH: str = Field("logs/app.log", env="LOG_FILE_PATH")
    MAX_CONCURRENT_TASKS: int = Field(2, env="MAX_CONCURRENT_TASKS")
    KEEP_ALIVE_INTERVAL_SECONDS: int = Field(30, env="KEEP_ALIVE_INTERVAL_SECONDS")
    GITHUB_API_BASE: str = Field("https://api.github.com", env="GITHUB_API_BASE")
    GITHUB_PAGES_BASE: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
if not settings.GITHUB_PAGES_BASE:
    settings.GITHUB_PAGES_BASE = f"https://{settings.GITHUB_USERNAME}.github.io"

# ------------------------- Logging -------------------------
os.makedirs(os.path.dirname(settings.LOG_FILE_PATH), exist_ok=True)
logger = logging.getLogger("task_receiver")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(settings.LOG_FILE_PATH, mode="a", encoding="utf-8")
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(fmt)
file_handler.setFormatter(fmt)
logger.handlers = []
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.propagate = False

def flush_logs():
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        for h in logger.handlers:
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        pass

# ------------------------- Models -------------------------
class Attachment(BaseModel):
    name: str
    url: str  # data URI or http(s) url

class TaskRequest(BaseModel):
    task: str
    email: str
    round: int
    brief: str
    evaluation_url: str
    nonce: str
    secret: str
    attachments: List[Attachment] = []
    checks: List[str] = []  # Evaluation criteria that will test the generated code

# ------------------------- App & Globals -------------------------
app = FastAPI(title="Automated Task Receiver & Processor", description="LLM-driven code generation and deployment")
background_tasks_list: List[asyncio.Task] = []
task_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)
last_received_task: Optional[dict] = None
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# ------------------------- Utility -------------------------
def verify_secret(secret_from_request: str) -> bool:
    return secret_from_request == settings.STUDENT_SECRET

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def remove_local_path(path: str):
    if not os.path.exists(path):
        return
    def onerror(func, path_arg, exc_info):
        try:
            os.chmod(path_arg, stat.S_IWUSR)
            func(path_arg)
        except Exception as exc:
            logger.exception(f"Failed in rmtree on {path_arg}: {exc}")
            raise
    logger.info(f"[CLEANUP] Removing local directory: {path}")
    shutil.rmtree(path, onerror=onerror)
    flush_logs()

# ------------------------- Attachment helpers -------------------------
def is_image_data_uri(data_uri: str) -> bool:
    if not data_uri or not data_uri.startswith("data:"):
        return False
    return re.search(r"data:image/[^;]+;base64,", data_uri, re.IGNORECASE) is not None

def data_uri_to_gemini_part(data_uri: str) -> Optional[dict]:
    if not data_uri or not data_uri.startswith("data:"):
        return None
    match = re.search(r"data:(?P<mime_type>[^;]+);base64,(?P<base64_data>.*)", data_uri, re.IGNORECASE)
    if not match:
        return None
    mime_type = match.group('mime_type')
    base64_data = match.group('base64_data')
    if not mime_type.startswith("image/"):
        return None
    return {"inlineData": {"data": base64_data, "mimeType": mime_type}}

async def attachment_to_gemini_part(attachment_url: str) -> Optional[dict]:
    if not attachment_url:
        return None
    if attachment_url.startswith("data:"):
        return data_uri_to_gemini_part(attachment_url)
    if attachment_url.startswith(("http://", "https://")):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(attachment_url)
                resp.raise_for_status()
                mime = resp.headers.get("Content-Type", "")
                if not mime.startswith("image/"):
                    logger.info(f"[ATTACHMENT] Skipping non-image MIME: {mime}")
                    return None
                b64 = base64.b64encode(resp.content).decode("utf-8")
                return {"inlineData": {"data": b64, "mimeType": mime}}
        except Exception as e:
            logger.warning(f"[ATTACHMENT] Failed to fetch/encode attachment {attachment_url}: {e}")
            return None
    return None

# ------------------------- Filesystem Save Helpers -------------------------
async def save_generated_files_locally(task_id: str, files: dict) -> str:
    base_dir = os.path.join(os.getcwd(), "generated_tasks")
    task_dir = os.path.join(base_dir, task_id)
    safe_makedirs(task_dir)
    logger.info(f"[LOCAL_SAVE] Saving generated files to: {task_dir}")
    for filename, content in files.items():
        file_path = os.path.join(task_dir, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"   -> Saved: {filename} (bytes: {len(content)})")
        except Exception as e:
            logger.exception(f"Failed to save generated file {filename}: {e}")
            raise
    flush_logs()
    return task_dir

async def save_attachments_locally(task_dir: str, attachments: List[Attachment]) -> List[str]:
    saved_files = []
    logger.info(f"[ATTACHMENTS] Processing {len(attachments)} attachments for {task_dir}")
    async with httpx.AsyncClient(timeout=30) as client:
        for attachment in attachments:
            filename = attachment.name
            url = attachment.url
            file_bytes = None
            if not filename or not url:
                logger.warning(f"Skipping invalid attachment entry: {filename}")
                continue
            try:
                if url.startswith("data:"):
                    m = re.search(r"base64,(.*)", url, re.IGNORECASE)
                    if m:
                        file_bytes = base64.b64decode(m.group(1))
                elif url.startswith(("http://", "https://")):
                    resp = await client.get(url)
                    resp.raise_for_status()
                    file_bytes = resp.content
                if file_bytes is None:
                    logger.warning(f"No content for attachment: {filename}")
                    continue
                file_path = os.path.join(task_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(file_bytes)
                logger.info(f"   -> Saved Attachment: {filename} (bytes: {len(file_bytes)})")
                saved_files.append(filename)
            except Exception as e:
                logger.exception(f"Failed to save attachment {filename}: {e}")
    flush_logs()
    return saved_files

# ------------------------- GitHub helpers -------------------------
async def setup_local_repo(local_path: str, repo_name: str, repo_url_auth: str, repo_url_http: str, round_index: int) -> git.Repo:
    github_token = settings.GITHUB_TOKEN
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    async with httpx.AsyncClient(timeout=45) as client:
        try:
            if round_index == 1:
                logger.info(f"[GIT] R1: Creating remote repo '{repo_name}'")
                payload = {"name": repo_name, "private": False, "auto_init": True}
                resp = await client.post(f"{settings.GITHUB_API_BASE}/user/repos", json=payload, headers=headers)
                resp.raise_for_status()
                repo = git.Repo.init(local_path)
                repo.create_remote('origin', repo_url_auth)
                logger.info("[GIT] Local repo initialized")
            else:
                logger.info(f"[GIT] R{round_index}: Cloning {repo_url_http}")
                repo = git.Repo.clone_from(repo_url_auth, local_path)
                logger.info("[GIT] Cloned repo")
            flush_logs()
            return repo
        except httpx.HTTPStatusError as e:
            logger.exception(f"GitHub API error: {getattr(e.response, 'text', '')}")
            raise
        except git.GitCommandError as e:
            logger.exception(f"Git command error: {e}")
            raise

async def commit_and_publish(repo: git.Repo, task_id: str, round_index: int, repo_name: str) -> dict:
    github_username = settings.GITHUB_USERNAME
    github_token = settings.GITHUB_TOKEN
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    repo_url_http = f"https://github.com/{github_username}/{repo_name}"
    async with httpx.AsyncClient(timeout=45) as client:
        try:

            # Log all files in the repo directory before commit
            repo_dir = repo.working_tree_dir
            all_files = []
            for root, dirs, files in os.walk(repo_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), repo_dir)
                    all_files.append(rel_path)
            logger.info(f"[GIT] Files in repo before commit: {all_files}")

            # Robustness: Check that all expected files (attachments and generated) are present
            task_dir = repo_dir
            missing_files = []
            for fname in os.listdir(task_dir):
                try:
                    _ = os.stat(os.path.join(task_dir, fname))
                except Exception as e:
                    logger.warning(f"[GIT] Could not stat file {fname}: {e}")
            # (Optional) You can add more checks here if you want to enforce presence of specific files

            repo.git.add(A=True)
            commit_message = f"Task {task_id} - Round {round_index}: automated update"
            repo.index.commit(commit_message)
            commit_sha = repo.head.object.hexsha
            logger.info(f"[GIT] Committed: {commit_sha}")
            repo.git.branch('-M', 'main')
            repo.git.push('--set-upstream', 'origin', 'main', force=True)
            logger.info("[GIT] Pushed to origin/main")

            # Configure GitHub Pages with retries
            pages_api_url = f"{settings.GITHUB_API_BASE}/repos/{github_username}/{repo_name}/pages"
            pages_payload = {"source": {"branch": "main", "path": "/"}}
            pages_max_retries = 5
            pages_base_delay = 3
            for attempt in range(pages_max_retries):
                try:
                    pages_response = await client.get(pages_api_url, headers=headers)
                    is_configured = (pages_response.status_code == 200)
                    if is_configured:
                        await client.put(pages_api_url, json=pages_payload, headers=headers)
                    else:
                        await client.post(pages_api_url, json=pages_payload, headers=headers)
                    logger.info("[GIT] Pages configured")
                    break
                except httpx.HTTPStatusError as e:
                    text = getattr(e.response, "text", "")
                    if e.response.status_code == 422 and "main branch must exist" in text and attempt < pages_max_retries - 1:
                        delay = pages_base_delay * (2 ** attempt)
                        logger.warning(f"[GIT] Pages timing issue, retrying in {delay}s")
                        await asyncio.sleep(delay)
                        continue
                    logger.exception(f"[GIT] Pages configuration failed: {text}")
                    raise

            await asyncio.sleep(5)  # allow pages to deploy
            pages_url = f"{settings.GITHUB_PAGES_BASE}/{repo_name}/"
            flush_logs()
            return {"repo_url": repo_url_http, "commit_sha": commit_sha, "pages_url": pages_url}
        except git.GitCommandError as e:
            logger.exception("Git operation failed during deployment.")
            raise
        except httpx.HTTPStatusError as e:
            logger.exception("GitHub API error during deployment.")
            raise

# ------------------------- Gemini / LLM helpers -------------------------
async def call_gemini_api(contents: list, system_prompt: str, response_schema: dict, max_retries: int = 3, timeout: int = 60) -> dict:
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }
    base_delay = 1
    for attempt in range(max_retries):
        try:
            if not settings.GEMINI_API_KEY:
                raise Exception("GEMINI_API_KEY not configured.")
            url = f"{GEMINI_API_URL}?key={settings.GEMINI_API_KEY}"
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
                resp.raise_for_status()
                result = resp.json()
                candidates = result.get("candidates", [])
                if not candidates:
                    raise ValueError("No candidates in LLM response")
                content_parts = candidates[0].get("content", {}).get("parts", [])
                if not content_parts:
                    raise ValueError("No content parts in candidate")
                json_text = content_parts[0].get("text")
                return json.loads(json_text)
        except httpx.HTTPStatusError as e:
            logger.warning(f"[GEMINI] HTTP error attempt {attempt+1}: {e}")
        except (httpx.RequestError, KeyError, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[GEMINI] Processing error attempt {attempt+1}: {e}")
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            logger.info(f"[GEMINI] Retrying in {delay}s...")
            await asyncio.sleep(delay)
    raise Exception("LLM generation failed after retries")

# ------------------------- Round 2 surgical update (Base.py style) -------------------------
async def call_llm_round2_surgical_update(task_id: str, brief: str, existing_index_html: str, checks: List[str] = None) -> dict:
    system_prompt = (
        "You are performing a SURGICAL CODE UPDATE. This is NOT a rewrite - you're FIXING/ENHANCING existing working code.\n\n"
        
        "🔒 IRON-CLAD PRESERVATION RULES (NEVER VIOLATE):\n"
        "1. If Papa Parse is in <head>, KEEP IT EXACTLY - don't remove, don't change version\n"
        "2. If CSV parsing uses Papa.parse(file, {download:true}), PRESERVE IT EXACTLY\n"
        "3. If DOMContentLoaded wrapper exists, KEEP IT - don't unwrap code\n"
        "4. If any library is loaded in <head>, DON'T REMOVE unless explicitly told\n"
        "5. If event handlers work, DON'T MODIFY unless required by new checks\n"
        "6. If calculations/logic work, DON'T REWRITE unless explicitly broken\n"
        "7. If element IDs exist and work, DON'T RENAME them\n"
        "8. If CSS/styling works, DON'T BREAK layout\n\n"
        
        "⚠️ STRICTLY BANNED OPERATIONS (will cause immediate failure):\n"
        "❌ Removing Papa Parse library from <head>\n"
        "❌ Changing Papa.parse({download:true, complete:...}) to fetch().then()\n"
        "❌ Removing working CSV data processing logic\n"
        "❌ Breaking existing event handlers or listeners\n"
        "❌ Removing DOMContentLoaded wrapper\n"
        "❌ Deleting working calculation functions\n"
        "❌ Removing error handling that exists\n"
        "❌ Breaking responsive design or CSS that works\n\n"
        
        "✅ ALLOWED OPERATIONS ONLY:\n"
        "• Add new elements with required IDs (if checks demand)\n"
        "• Add missing libraries to <head> (if checks require)\n"
        "• Update/add <title> tag (if checks require)\n"
        "• Add new functions without modifying working ones\n"
        "• Enhance styling WITHOUT breaking existing layout\n"
        "• Add error handling if missing\n"
        "• Fix genuinely broken logic (if identified by checks)\n"
        "• Add new features explicitly requested in brief\n\n"
        
        "📋 SURGICAL UPDATE PROCESS:\n"
        "Step 1: Read existing code - what ALREADY WORKS?\n"
        "Step 2: Read new requirements - what's MISSING or BROKEN?\n"
        "Step 3: Identify MINIMAL changes needed\n"
        "Step 4: Make ONLY those changes\n"
        "Step 5: Verify you didn't break anything that worked\n"
        "Step 6: Verify all new checks will pass\n\n"
        
        "⚠️ EVALUATION CRITERIA COMPLIANCE:\n"
        "Each check is a NEW REQUIREMENT you must fulfill:\n\n"
        
        "1. ELEMENT ID CHECK:\n"
        "   'element with id=\"xyz\" exists' → Add <div id=\"xyz\"></div> if missing\n"
        "   ✅ Add element if missing\n"
        "   ❌ Don't remove existing elements to make room\n\n"
        
        "2. TITLE CHECK:\n"
        "   'document.title is \"XYZ\"' → Update <title>XYZ</title>\n"
        "   ✅ Change only the <title> tag\n"
        "   ❌ Don't modify anything else in <head>\n\n"
        
        "3. LIBRARY CHECK:\n"
        "   'Library ABC is loaded' → Add <script src=\"...\"></script> to <head>\n"
        "   ✅ Add missing library\n"
        "   ❌ Don't remove or modify existing libraries\n\n"
        
        "4. CALCULATION CHECK:\n"
        "   'sum calculation is correct' → Fix/add calculation logic\n"
        "   ✅ Add/fix the specific calculation\n"
        "   ❌ Don't rewrite entire data processing pipeline\n\n"
        
        "5. CONTENT CHECK:\n"
        "   'page contains \"Welcome\"' → Add <h1>Welcome</h1> if missing\n"
        "   ✅ Add missing content\n"
        "   ❌ Don't remove existing content\n\n"
        
        "6. INTERACTION CHECK:\n"
        "   'button triggers action' → Add button + event listener if missing\n"
        "   ✅ Add missing button/handler\n"
        "   ❌ Don't break existing buttons/handlers\n\n"
        
        "💡 CSV PRESERVATION (MOST CRITICAL):\n"
        "If existing code has Papa Parse CSV handling:\n"
        "  <script src=\"...papaparse...\"></script>\n"
        "  Papa.parse('data.csv', {download: true, complete: function(results) {...}})\n\n"
        
        "→ PRESERVE IT EXACTLY - This is the CORRECT pattern\n"
        "→ NEVER change to fetch() - that's a DOWNGRADE\n"
        "→ Only modify if genuinely broken (e.g., wrong filename)\n\n"
        
        "🧪 PRE-SUBMISSION VERIFICATION:\n"
        "Ask yourself these questions before responding:\n\n"
        
        "□ Did I keep ALL existing libraries in <head>?\n"
        "□ Did I keep Papa Parse if it existed?\n"
        "□ Did I keep Papa.parse({download:true}) pattern if it existed?\n"
        "□ Did I keep DOMContentLoaded wrapper if it existed?\n"
        "□ Did I keep all working event handlers?\n"
        "□ Did I keep all working calculation logic?\n"
        "□ Did I only ADD what's required by new checks?\n"
        "□ Did I test each new check mentally - will it pass?\n"
        "□ Did I verify existing features still work?\n"
        "□ Is this truly a SURGICAL update (not a rewrite)?\n\n"
        
        "If ANY answer is 'no' → REVISE before responding!\n\n"
        
        "⚠️ SAFETY GUIDELINES:\n"
        "• Return COMPLETE files (not diffs)\n"
        "• If README.md doesn't need changes, return existing content\n"
        "• If LICENSE doesn't need changes, return existing content\n"
        "• When in doubt about removing something → DON'T remove it\n"
        "• When in doubt about changing something → DON'T change it\n"
        "• ONLY make changes you're 100% certain are required\n\n"
        
        "🎯 SUCCESS CRITERIA:\n"
        "Your update is successful if:\n"
        "1. ALL new evaluation checks pass\n"
        "2. ALL existing features still work\n"
        "3. Code is measurably BETTER (not just different)\n"
        "4. Changes are MINIMAL and TARGETED\n"
        "5. Nothing working was broken in the process"
    )
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "index.html": {"type": "STRING"},
            "README.md": {"type": "STRING"},
            "LICENSE": {"type": "STRING"},
        },
        "required": ["index.html", "README.md", "LICENSE"]
    }
    
    # Build structured update prompt
    prompt_parts = [
        "=" * 80,
        "SURGICAL UPDATE TASK",
        "=" * 80,
        "You are updating existing working code. Make MINIMAL changes only.",
        "",
        "=" * 80,
        "NEW REQUIREMENTS",
        "=" * 80,
        brief,
        ""
    ]
    
    # Add evaluation criteria if present
    if checks:
        prompt_parts.extend([
            "=" * 80,
            "🎯 EVALUATION CRITERIA (MUST PASS)",
            "=" * 80,
            "After your update, ALL these checks must pass:",
            ""
        ])
        for idx, check in enumerate(checks, 1):
            prompt_parts.append(f"CHECK {idx}: {check}")
        prompt_parts.extend([
            "",
            "⚠️ Action Items:",
            "• Add any missing elements with exact IDs",
            "• Include any missing libraries via CDN",
            "• Fix any broken functionality",
            "• Preserve everything that already works",
            ""
        ])
    
    prompt_parts.extend([
        "=" * 80,
        "EXISTING CODE (DO NOT BREAK THIS)",
        "=" * 80,
        existing_index_html,
        "",
        "=" * 80,
        "UPDATE INSTRUCTIONS",
        "=" * 80,
        "1. Analyze what currently works",
        "2. Identify what needs to change for new requirements",
        "3. Make ONLY necessary changes",
        "4. Ensure all evaluation checks pass",
        "5. Do NOT remove or break existing features",
        "",
        "Return complete JSON with 'index.html', 'README.md', 'LICENSE'.",
        "If README/LICENSE don't need changes, return their current content.",
        "=" * 80
    ])
    
    prompt = "\n".join(prompt_parts)

    contents = [{"parts": [{"text": prompt}]}]

    try:
        result = await call_gemini_api(contents=contents, system_prompt=system_prompt, response_schema=response_schema, max_retries=4, timeout=90)
    except Exception as e:
        logger.exception(f"[ROUND2] LLM call failed: {e}")
        # Fallback: return existing index.html and placeholders for readme/license
        return {"index.html": existing_index_html or "<!-- original index.html preserved due to LLM failure -->",
                "README.md": "", "LICENSE": ""}

    # Safety checks (Safe Mode)
    new_html = (result.get("index.html") or "").strip()
    if not new_html:
        logger.warning("[SAFE] LLM returned empty index.html — reverting to existing.")
        result["index.html"] = existing_index_html
    else:
        # If LLM output is grossly shorter than original (possible destructive rewrite), reject it.
        try:
            orig_len = len(existing_index_html or "")
            new_len = len(new_html)
            if orig_len > 0 and new_len < max(200, int(orig_len * 0.3)):  # threshold: not less than 30% (and at least 200 chars)
                logger.warning("[SAFE] LLM index.html appears destructive (too small). Reverting to existing.")
                result["index.html"] = existing_index_html
        except Exception:
            # if anything goes wrong in safety checks, revert
            result["index.html"] = existing_index_html

    # Ensure README and LICENSE exist (if LLM didn't return them)
    result["README.md"] = result.get("README.md") or ""
    result["LICENSE"] = result.get("LICENSE") or ""
    return result

# ------------------------- Notifier -------------------------
async def notify_evaluation_server(evaluation_url: str, email: str, task_id: str, round_index: int, nonce: str, repo_url: str, commit_sha: str, pages_url: str) -> bool:
    payload = {
        "email": email,
        "task": task_id,
        "round": round_index,
        "nonce": nonce,
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "pages_url": pages_url
    }
    max_retries = 3
    base_delay = 1
    logger.info(f"[NOTIFY] Notifying evaluation server at {evaluation_url}")
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(evaluation_url, json=payload)
                resp.raise_for_status()
                logger.info(f"[NOTIFY] Notification succeeded: {resp.status_code}")
                flush_logs()
                return True
        except httpx.HTTPStatusError as e:
            logger.warning(f"[NOTIFY] HTTP error attempt {attempt+1}: {e}")
        except httpx.RequestError as e:
            logger.warning(f"[NOTIFY] Request error attempt {attempt+1}: {e}")
        if attempt < max_retries - 1:
            await asyncio.sleep(base_delay * (2 ** attempt))
    logger.error("[NOTIFY] Failed to notify evaluation server after retries.")
    flush_logs()
    return False

# ------------------------- Main orchestration -------------------------
async def generate_files_and_deploy(task_data: TaskRequest):
    acquired = False
    try:
        await task_semaphore.acquire()
        acquired = True
        logger.info(f"[PROCESS START] Task: {task_data.task} Round: {task_data.round}")
        flush_logs()

        task_id = task_data.task
        email = task_data.email
        round_index = task_data.round
        brief = task_data.brief
        evaluation_url = task_data.evaluation_url
        nonce = task_data.nonce
        attachments = task_data.attachments or []

        repo_name = task_id.replace(" ", "-").lower()
        github_username = settings.GITHUB_USERNAME
        github_token = settings.GITHUB_TOKEN
        repo_url_auth = f"https://{github_username}:{github_token}@github.com/{github_username}/{repo_name}.git"
        repo_url_http = f"https://github.com/{github_username}/{repo_name}"

        base_dir = os.path.join(os.getcwd(), "generated_tasks")
        local_path = os.path.join(base_dir, task_id)

        # Cleanup local path
        if os.path.exists(local_path):
            try:
                remove_local_path(local_path)
            except Exception as e:
                logger.exception(f"Cleanup failed for {local_path}: {e}")
                raise
        safe_makedirs(local_path)

        # Setup repo (init or clone)
        repo = await setup_local_repo(local_path, repo_name, repo_url_auth, repo_url_http, round_index)

        # --- Prepare attachment data for LLM ---
        image_parts = []
        for attachment in attachments:
            part = await attachment_to_gemini_part(attachment.url)
            if part:
                image_parts.append(part)

        # Build explicit file reference list for LLM
        attachment_descriptions = ""
        if attachments:
            attachment_descriptions = "\n📎 ATTACHED FILES (all in same directory as index.html):\n"
            for att in attachments:
                file_ext = att.name.split('.')[-1].lower() if '.' in att.name else ''
                
                if file_ext == 'csv':
                    attachment_descriptions += (
                        f"\n• {att.name} (CSV file)\n"
                        f"  ✅ MUST use Papa Parse library:\n"
                        f"  <script src=\"https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js\"></script>\n"
                        f"  \n"
                        f"  ✅ MUST use this pattern:\n"
                        f"  Papa.parse('{att.name}', {{\n"
                        f"    download: true,  // Papa fetches the file automatically\n"
                        f"    header: true,    // First row = column names\n"
                        f"    dynamicTyping: true,  // Auto-convert numbers\n"
                        f"    complete: function(results) {{\n"
                        f"      console.log('Parsed:', results.data);\n"
                        f"      // Your data processing here\n"
                        f"    }},\n"
                        f"    error: function(err) {{ console.error('Error:', err); }}\n"
                        f"  }});\n"
                    )
                elif file_ext in ['json']:
                    attachment_descriptions += (
                        f"\n• {att.name} (JSON file)\n"
                        f"  Use: fetch('{att.name}').then(r => r.json()).then(data => {{ /* use data */ }})\n"
                    )
                elif file_ext in ['jpg', 'jpeg', 'png', 'gif', 'svg', 'webp']:
                    attachment_descriptions += (
                        f"\n• {att.name} (Image file)\n"
                        f"  Use: <img src=\"{att.name}\" alt=\"Description\">\n"
                    )
                elif file_ext in ['md', 'markdown']:
                    attachment_descriptions += (
                        f"\n• {att.name} (Markdown file)\n"
                        f"  1. Include: <script src=\"https://cdn.jsdelivr.net/npm/marked/marked.min.js\"></script>\n"
                        f"  2. Use: fetch('{att.name}').then(r => r.text()).then(md => marked.parse(md))\n"
                    )
                elif file_ext in ['txt']:
                    attachment_descriptions += (
                        f"\n• {att.name} (Text file)\n"
                        f"  Use: fetch('{att.name}').then(r => r.text()).then(text => {{ /* use text */ }})\n"
                    )
                else:
                    attachment_descriptions += f"  • {att.name} (file)\n"
            
            attachment_descriptions += (
                "\n💡 CRITICAL FILE HANDLING RULES:\n"
                "• All files (CSV, JSON, images, etc.) must be saved in the SAME directory as index.html\n"
                "• All file references in HTML/JS must use the EXACT filename (case-sensitive)\n"
                "• Use only relative paths (e.g., 'data.csv', 'logo.png')\n"
                "• Do NOT use subfolders or absolute paths\n"
                "• If a file is missing, the app will not work on GitHub Pages\n"
                "• For CSV: Use Papa Parse with download:true, and reference the file as 'filename.csv'\n"
                "• For JSON: Use fetch('filename.json')\n"
                "• For images: Use <img src='filename.ext'>\n"
                "• For markdown: Use fetch('filename.md') and marked.js\n"
                "• For text: Use fetch('filename.txt')\n"
                "• If you reference a file, it MUST be present in the repo and committed\n"
            )

        # --- Round 1: Full generation ---
        if round_index == 1:
            logger.info("[WORKFLOW] Round 1: full generation")

            # Build structured prompt with clear sections
            prompt_parts = [
                "=" * 80,
                "TASK BRIEF",
                "=" * 80,
                brief,
                ""
            ]
            
            # Add attachments section if present
            if attachment_descriptions:
                prompt_parts.extend([
                    "=" * 80,
                    "ATTACHED FILES",
                    "=" * 80,
                    attachment_descriptions.strip(),
                    ""
                ])
            
            # Add evaluation criteria (CRITICAL for passing checks)
            if hasattr(task_data, 'checks') and task_data.checks:
                prompt_parts.extend([
                    "=" * 80,
                    "🎯 EVALUATION CRITERIA (MANDATORY - YOUR CODE WILL BE TESTED)",
                    "=" * 80,
                    "Your application MUST satisfy ALL these checks to pass:",
                    ""
                ])
                for idx, check in enumerate(task_data.checks, 1):
                    prompt_parts.append(f"CHECK {idx}: {check}")
                prompt_parts.extend([
                    "",
                    "⚠️ CRITICAL REMINDERS:",
                    "• Element IDs must match EXACTLY (case-sensitive)",
                    "• document.title must be set in <title> tag",
                    "• Required libraries must be loaded via CDN in <head> section",
                    "• Papa Parse MUST be loaded before any CSV operations",
                    "• Use Papa.parse() with download:true for CSV files (not fetch)",
                    "• Calculations must produce exact expected results",
                    "• Event handlers must be properly attached",
                    "• Wrap JavaScript in DOMContentLoaded listener",
                    ""
                ])
            
            prompt_parts.extend([
                "=" * 80,
                "REQUIREMENTS SUMMARY",
                "=" * 80,
                "✅ Create fully functional HTML application",
                "✅ Include ALL required libraries via CDN",
                "✅ Handle all attached files correctly",
                "✅ Ensure ALL evaluation checks pass",
                "✅ Add proper error handling",
                "✅ Make responsive and user-friendly",
                "=" * 80
            ])
            
            enriched_brief = "\n".join(prompt_parts)

            system_prompt = (
                "You are an expert full-stack web developer. Create a complete, working web application that PASSES ALL CHECKS.\n"
                "Return JSON with keys: 'index.html', 'README.md', 'LICENSE'.\n\n"
                
                "🎯 ABSOLUTE REQUIREMENTS (NON-NEGOTIABLE):\n"
                "1. Build a FULLY FUNCTIONAL application - every feature must WORK\n"
                "2. Meet EVERY evaluation criterion EXACTLY as specified\n"
                "3. Handle edge cases, errors, and empty data gracefully\n"
                "4. Test your code mentally against EVERY check before responding\n"
                "5. If unsure about a requirement, IMPLEMENT IT - don't skip it\n\n"
                
                "⚠️ EVALUATION CRITERIA - MANDATORY COMPLIANCE:\n"
                "Each check is a REQUIREMENT that MUST pass. Common patterns:\n\n"
                
                "1. ELEMENT ID CHECK:\n"
                "   Check: 'element with id=\"total-sales\" exists'\n"
                "   Required: <div id=\"total-sales\"></div> (or any tag with EXACT id)\n"
                "   ❌ FAIL: Using different id, typo, or missing element\n"
                "   ✅ PASS: Element with EXACT matching id exists\n\n"
                
                "2. DOCUMENT TITLE CHECK:\n"
                "   Check: 'document.title is \"Sales Dashboard\"'\n"
                "   Required: <head><title>Sales Dashboard</title></head>\n"
                "   ❌ FAIL: Missing <title>, wrong text, extra characters\n"
                "   ✅ PASS: <title> tag with EXACT matching text\n\n"
                
                "3. LIBRARY/DEPENDENCY CHECK:\n"
                "   Check: 'Papa Parse library is loaded'\n"
                "   Required: <script src=\"https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js\"></script>\n"
                "   ❌ FAIL: Missing library, wrong version, loaded after use\n"
                "   ✅ PASS: Library loaded in <head> before any script that uses it\n\n"
                
                "4. CALCULATION/LOGIC CHECK:\n"
                "   Check: 'sum of sales column is calculated correctly'\n"
                "   Required: Parse CSV, sum numeric values, display result\n"
                "   ❌ FAIL: Wrong formula, missing calculation, hardcoded value\n"
                "   ✅ PASS: Dynamic calculation that works with any valid data\n\n"
                
                "5. CONTENT/TEXT CHECK:\n"
                "   Check: 'page contains heading \"Welcome Dashboard\"'\n"
                "   Required: <h1>Welcome Dashboard</h1> (or h2-h6)\n"
                "   ❌ FAIL: Missing text, typo, wrong case, hidden element\n"
                "   ✅ PASS: Visible text matching exactly\n\n"
                
                "6. INTERACTION/BUTTON CHECK:\n"
                "   Check: 'button with id=\"calculate-btn\" triggers calculation'\n"
                "   Required: <button id=\"calculate-btn\"></button> + event listener\n"
                "   ❌ FAIL: Missing button, wrong id, no event handler\n"
                "   ✅ PASS: Button exists with working click handler\n\n"
                
                "💡 IMPLEMENTATION GUIDELINES:\n\n"
                
                "**File Handling Priority:**\n"
                "• CSV files → ALWAYS use Papa Parse with download:true (see CSV section)\n"
                "• JSON files → fetch() + response.json()\n"
                "• Images → <img src=\"filename.ext\" alt=\"description\">\n"
                "• Markdown → marked.js library via CDN + fetch()\n"
                "• Text files → fetch() + response.text()\n"
                "• Multiple files → Process ALL of them, don't skip any\n\n"
                
                "**CSV HANDLING (MOST CRITICAL):**\n"
                "⚠️ #1 MISTAKE: Using fetch() + Papa.parse(text) - DON'T DO THIS!\n\n"
                
                "✅ CORRECT METHOD (copy this exactly):\n"
                "Step 1 - Load Papa Parse in <head>:\n"
                "  <script src=\"https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js\"></script>\n\n"
                
                "Step 2 - Wait for DOM and verify Papa is loaded:\n"
                "  document.addEventListener('DOMContentLoaded', function() {\n"
                "    if (typeof Papa === 'undefined') {\n"
                "      console.error('Papa Parse not loaded!');\n"
                "      document.body.innerHTML = '<h1>Error: Library failed to load</h1>';\n"
                "      return;\n"
                "    }\n"
                "    loadData();\n"
                "  });\n\n"
                
                "Step 3 - Use Papa.parse() with download:true:\n"
                "  function loadData() {\n"
                "    Papa.parse('data.csv', {  // Use exact filename from attachments\n"
                "      download: true,  // ← Papa fetches the file automatically\n"
                "      header: true,    // First row = column names\n"
                "      dynamicTyping: true,  // Auto-convert numbers\n"
                "      skipEmptyLines: true,\n"
                "      complete: function(results) {\n"
                "        if (results.errors.length > 0) {\n"
                "          console.error('Parse errors:', results.errors);\n"
                "        }\n"
                "        console.log('Parsed data:', results.data);\n"
                "        processData(results.data);  // Your logic here\n"
                "      },\n"
                "      error: function(error) {\n"
                "        console.error('Failed to load CSV:', error);\n"
                "        document.getElementById('result').textContent = 'Error loading data';\n"
                "      }\n"
                "    });\n"
                "  }\n\n"
                
                "❌ NEVER DO THIS:\n"
                "  fetch('data.csv')\n"
                "    .then(r => r.text())\n"
                "    .then(csv => Papa.parse(csv))  // ← WRONG PATTERN\n\n"
                
                "✅ ALWAYS DO THIS:\n"
                "  Papa.parse('data.csv', {download: true, complete: ...})  // ← RIGHT PATTERN\n\n"
                
                "**Required Libraries (load in <head> via CDN):**\n"
                "• CSS Framework: Tailwind CSS (unless Bootstrap/other specified)\n"
                "  <script src=\"https://cdn.tailwindcss.com\"></script>\n"
                "• CSV Parsing: Papa Parse\n"
                "  <script src=\"https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js\"></script>\n"
                "• Charts: Chart.js\n"
                "  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n"
                "• Markdown: marked.js\n"
                "  <script src=\"https://cdn.jsdelivr.net/npm/marked/marked.min.js\"></script>\n"
                "• Icons: Font Awesome\n"
                "  <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css\">\n\n"
                
                "**Code Quality Standards:**\n"
                "• Modern JavaScript (ES6+: const/let, arrow functions, template literals)\n"
                "• Responsive design (works on mobile, tablet, desktop)\n"
                "• Error handling for ALL async operations (try-catch, error callbacks)\n"
                "• User feedback (loading spinners, success messages, error alerts)\n"
                "• Accessibility (semantic HTML, alt text, ARIA labels)\n"
                "• Clean code (meaningful variable names, comments for complex logic)\n\n"
                
                "**File Paths (CRITICAL):**\n"
                "• All attachments are in SAME directory as index.html\n"
                "• Use relative paths: 'filename.csv' or './filename.csv'\n"
                "• Match filenames EXACTLY (case-sensitive)\n"
                "• Example: If attachment is 'Sales_Data.csv', use 'Sales_Data.csv' (not 'sales_data.csv')\n\n"
                
                "**Documentation Requirements:**\n"
                "• README.md: Include project title, description, features, how to use\n"
                "• LICENSE: Complete MIT License with year 2025\n\n"
                
                "🚫 COMMON MISTAKES THAT CAUSE FAILURES:\n"
                "1. ❌ Using Papa Parse without loading library first\n"
                "2. ❌ Using fetch() instead of Papa.parse({download:true}) for CSV\n"
                "3. ❌ Running code before DOM is ready (missing DOMContentLoaded)\n"
                "4. ❌ Missing required libraries in <head> section\n"
                "5. ❌ Wrong element IDs (typos, case mismatch)\n"
                "6. ❌ Missing or incorrect <title> tag\n"
                "7. ❌ Wrong file paths or filenames\n"
                "8. ❌ Missing error handling (app breaks on bad data)\n"
                "9. ❌ Hardcoded values instead of dynamic calculations\n"
                "10. ❌ Incomplete features (TODOs, placeholders, commented code)\n\n"
                
                "🧪 PRE-SUBMISSION CHECKLIST (MANDATORY):\n"
                "Before returning your code, verify each item:\n\n"
                
                "□ 1. Did I read ALL evaluation criteria?\n"
                "□ 2. Did I create elements with EXACT IDs from checks?\n"
                "□ 3. Did I set <title> tag to match document.title check?\n"
                "□ 4. Did I include ALL required libraries via CDN in <head>?\n"
                "□ 5. If CSV files exist, did I include Papa Parse?\n"
                "□ 6. If CSV files exist, did I use Papa.parse({download:true})?\n"
                "□ 7. Is JavaScript wrapped in DOMContentLoaded?\n"
                "□ 8. Are file paths relative and case-sensitive correct?\n"
                "□ 9. Did I implement ALL calculations/logic from requirements?\n"
                "□ 10. Did I add error handling for ALL async operations?\n"
                "□ 11. Does the app work with empty/invalid data?\n"
                "□ 12. Are there NO placeholders, TODOs, or incomplete features?\n"
                "□ 13. Will this code work immediately on deployment?\n"
                "□ 14. Can I mentally trace how EACH check will pass?\n\n"
                
                "If ANY checkbox is unchecked → FIX IT before responding!\n\n"
                
                "✅ FINAL VERIFICATION:\n"
                "Mentally simulate the evaluation:\n"
                "• Check 1: [Can this pass?] → [Why/how?]\n"
                "• Check 2: [Can this pass?] → [Why/how?]\n"
                "• ... for EVERY check\n\n"
                
                "Only return code when you're 100% confident ALL checks will pass."
            )

            response_schema = {
                "type": "OBJECT",
                "properties": {
                    "index.html": {"type": "STRING"},
                    "README.md": {"type": "STRING"},
                    "LICENSE": {"type": "STRING"},
                },
                "required": ["index.html", "README.md", "LICENSE"],
            }

            contents = []
            if image_parts:
                contents.append({"parts": image_parts + [{"text": enriched_brief}]})
            else:
                contents.append({"parts": [{"text": enriched_brief}]})

            # Generate with validation and retry
            max_attempts = 3 if task_data.checks and VALIDATION_AVAILABLE else 1
            generated = None
            
            for attempt in range(1, max_attempts + 1):
                logger.info(f"[GENERATION] Attempt {attempt}/{max_attempts}")
                
                try:
                    generated = await call_gemini_api(
                        contents=contents,
                        system_prompt=system_prompt,
                        response_schema=response_schema,
                        max_retries=4,
                        timeout=120,
                    )
                    
                    # Validate if checks are provided and validation is available
                    if task_data.checks and VALIDATION_AVAILABLE:
                        logger.info("[VALIDATION] Pre-deployment validation starting...")
                        validation_result = validate_generated_code(
                            html_content=generated.get("index.html", ""),
                            checks=task_data.checks,
                            attachments=[{"name": att.name} for att in attachments]
                        )
                        
                        logger.info(f"[VALIDATION] Score: {validation_result['score']:.1f}% | "
                                  f"Passed: {len(validation_result['passed'])} | "
                                  f"Failed: {len(validation_result['failed'])}")
                        
                        # If validation passes or it's the last attempt, accept it
                        if validation_result["valid"] or attempt == max_attempts:
                            if validation_result["valid"]:
                                logger.info(f"[GENERATION] ✅ Success on attempt {attempt}")
                            else:
                                logger.warning(f"[GENERATION] ⚠️ Validation failed on final attempt, proceeding anyway")
                                for failure in validation_result['failed'][:5]:  # Log first 5 failures
                                    logger.warning(f"  {failure}")
                            break
                        
                        # If validation failed and not last attempt, retry with feedback
                        logger.warning(f"[GENERATION] ❌ Attempt {attempt} failed validation, retrying...")
                        for failure in validation_result['failed']:
                            logger.warning(f"  {failure}")
                        
                        # Add comprehensive feedback to the prompt for next attempt
                        feedback_parts = [
                            "",
                            "=" * 80,
                            f"⚠️ VALIDATION FAILED - ATTEMPT {attempt}/{max_attempts}",
                            "=" * 80,
                            f"Your previous code failed {len(validation_result['failed'])} checks:",
                            ""
                        ]
                        
                        for i, failure in enumerate(validation_result['failed'], 1):
                            feedback_parts.append(f"{i}. ❌ {failure}")
                        
                        feedback_parts.extend([
                            "",
                            "🔧 REQUIRED FIXES:",
                            "• Add missing elements with EXACT IDs (case-sensitive)",
                            "• Include missing libraries in <head> via CDN",
                            "• Set correct <title> tag text",
                            "• Fix broken logic/calculations",
                            "• Verify all file references are correct",
                            "• If CSV files: Use Papa.parse({download:true}), NOT fetch()",
                            "• Ensure all event handlers are properly attached",
                            "• Add missing content/text that checks require",
                            "",
                            "🎯 CRITICAL REMINDERS:",
                            "• Read each failed check carefully",
                            "• Implement the EXACT requirement",
                            "• Don't assume - implement explicitly",
                            "• Test mentally: will this check pass now?",
                            "",
                            "✅ Generate CORRECTED code that passes ALL checks.",
                            "=" * 80,
                            ""
                        ])
                        
                        feedback = "\n".join(feedback_parts)
                        enriched_brief_with_feedback = enriched_brief + feedback
                        contents = []
                        if image_parts:
                            contents.append({"parts": image_parts + [{"text": enriched_brief_with_feedback}]})
                        else:
                            contents.append({"parts": [{"text": enriched_brief_with_feedback}]})
                    else:
                        # No validation, accept the result
                        logger.info("[GENERATION] ✅ Generated successfully (no validation)")
                        break
                        
                except Exception as e:
                    logger.exception(f"[GENERATION] Attempt {attempt} failed with error: {e}")
                    if attempt == max_attempts:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            if generated is None:
                raise Exception("Failed to generate code after all attempts")

        # --- Round 2+: Surgical Update ---
        else:
            logger.info("[WORKFLOW] Round 2+: surgical update (Base.py style). Loading existing index.html only.")
            existing_index_html = ""
            idx_path = os.path.join(local_path, "index.html")
            if os.path.exists(idx_path):
                try:
                    with open(idx_path, "r", encoding="utf-8") as f:
                        existing_index_html = f.read()
                    logger.info("[WORKFLOW] Read existing index.html for context.")
                except Exception as e:
                    logger.warning(f"[WORKFLOW] Could not read existing index.html: {e}")
                    existing_index_html = ""

            # Build structured brief for round 2 with attachments
            brief_parts = [brief]
            if attachment_descriptions:
                brief_parts.extend(["", attachment_descriptions.strip()])
            brief_with_attachments = "\n".join(brief_parts)
            
            generated = await call_llm_round2_surgical_update(
                task_id=task_id, 
                brief=brief_with_attachments, 
                existing_index_html=existing_index_html,
                checks=task_data.checks if hasattr(task_data, 'checks') else None
            )

            # Preserve README/LICENSE if LLM didn’t return them
            readme_path = os.path.join(local_path, "README.md")
            license_path = os.path.join(local_path, "LICENSE")
            if not generated.get("README.md") and os.path.exists(readme_path):
                with open(readme_path, "r", encoding="utf-8") as f:
                    generated["README.md"] = f.read()
            if not generated.get("LICENSE") and os.path.exists(license_path):
                with open(license_path, "r", encoding="utf-8") as f:
                    generated["LICENSE"] = f.read()

        # Save generated files locally
        await save_generated_files_locally(task_id, generated)

        # Save attachments into repo folder
        await save_attachments_locally(os.path.join(base_dir, task_id), attachments)

        # Commit and publish
        deployment_info = await commit_and_publish(repo, task_id, round_index, repo_name)

        # Delay notification by 15 seconds
        logger.info("[NOTIFY] Waiting 30 seconds before notifying evaluation server...")
        await asyncio.sleep(30)
        # Notify evaluation server
        await notify_evaluation_server(
            evaluation_url=evaluation_url,
            email=email,
            task_id=task_id,
            round_index=round_index,
            nonce=nonce,
            repo_url=deployment_info["repo_url"],
            commit_sha=deployment_info["commit_sha"],
            pages_url=deployment_info["pages_url"],
        )

        logger.info(f"[DEPLOYMENT] Success. Repo: {deployment_info['repo_url']} Pages: {deployment_info['pages_url']}")

    except Exception as exc:
        logger.exception(f"[CRITICAL FAILURE] Task {getattr(task_data, 'task', 'unknown')} failed: {exc}")
    finally:
        if acquired:
            task_semaphore.release()
        flush_logs()
        logger.info(
            f"[PROCESS END] Task: {getattr(task_data, 'task', 'unknown')} Round: {getattr(task_data, 'round', 'unknown')}"
        )


# ------------------------- Endpoint handlers -------------------------
def _task_done_callback(task: asyncio.Task):
    try:
        exc = task.exception()
        if exc:
            logger.error(f"[BACKGROUND TASK] Task finished with exception: {exc}")
            logger.exception(exc)
        else:
            logger.info("[BACKGROUND TASK] Task finished successfully.")
    except asyncio.CancelledError:
        logger.warning("[BACKGROUND TASK] Task was cancelled.")
    finally:
        flush_logs()

@app.post("/task", status_code=200)
async def receive_task(task_data: TaskRequest, request: Request):
    global last_received_task, background_tasks_list
    if not verify_secret(task_data.secret):
        logger.warning(f"Unauthorized attempt for task again {task_data.task} from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(status_code=401, detail="Unauthorized: Secret mismatch")

    last_received_task = {
        "task": task_data.task,
        "email": task_data.email,
        "round": task_data.round,
        "brief": (task_data.brief[:250] + "...") if len(task_data.brief) > 250 else task_data.brief,
        "time": datetime.utcnow().isoformat() + "Z"
    }

    bg_task = asyncio.create_task(generate_files_and_deploy(task_data))
    bg_task.add_done_callback(_task_done_callback)
    background_tasks_list.append(bg_task)

    logger.info(f"Received task {task_data.task}. Background processing started.")
    flush_logs()

    return JSONResponse(status_code=200, content={"status": "task", "message": f"Task {task_data.task} received and processing started."})

@app.get("/")
async def root():
    return {"message": "Task Receiver Service running. POST /task to submit."}

@app.get("/status")
async def get_status():
    global last_received_task, background_tasks_list
    if last_received_task:
        background_tasks_list[:] = [t for t in background_tasks_list if not t.done()]
        return {"last_received_task": last_received_task, "running_background_tasks": len(background_tasks_list)}
    return {"message": "Awaiting first task submission to /task"}

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}

@app.get("/logs")
async def get_logs(lines: int = Query(200, ge=1, le=5000)):
    path = settings.LOG_FILE_PATH
    if not os.path.exists(path):
        return PlainTextResponse("Log file not found.", status_code=404)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            buffer = bytearray()
            block_size = 1024
            blocks = 0
            # Read from end until we have enough or hit limit
            while file_size > 0 and len(buffer) < lines * 2000 and blocks < 1024:
                read_size = min(block_size, file_size)
                f.seek(file_size - read_size)
                buffer.extend(f.read(read_size))
                file_size -= read_size
                blocks += 1
            text = buffer.decode(errors="ignore").splitlines()
            last_lines = "\n".join(text[-lines:])
            return PlainTextResponse(last_lines)
    except Exception as e:
        logger.exception(f"Error reading log file: {e}")
        return PlainTextResponse(f"Error reading log file: {e}", status_code=500)

# ------------------------- Startup / Shutdown -------------------------
@app.on_event("startup")
async def startup_event():
    async def keep_alive():
        while True:
            try:
                logger.info("[running....] sever running")
                flush_logs()
            except Exception:
                pass
            await asyncio.sleep(settings.KEEP_ALIVE_INTERVAL_SECONDS)
    asyncio.create_task(keep_alive())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("[SHUTDOWN] Waiting for background tasks to finish (graceful shutdown)...")
    for t in background_tasks_list:
        if not t.done():
            try:
                t.cancel()
            except Exception:
                pass
    await asyncio.sleep(0.5)
    flush_logs()
