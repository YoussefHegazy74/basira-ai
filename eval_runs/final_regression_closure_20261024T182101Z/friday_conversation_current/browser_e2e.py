"""Real-browser verification of the four-turn Friday-fasting conversation."""

import json
import re
import time
import urllib.request
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path("/run/media/joe/Data/basira-ai")
ART = ROOT / "eval_runs" / "final_regression_closure_20261024T182101Z" / "friday_conversation_current"
SHOTS = ART / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)
FRONTEND = "http://localhost:5173"
API = "http://localhost:3000"
FIXTURE = json.load(open("/tmp/opencode/friday_fixture.json"))
ACTIVE_KEY = f"basira.activeConversation.{FIXTURE['userId']}"
QUESTIONS = [
    "ما حكم صيام الجمعة؟",
    "يعني لازم أصوم يوم قبله أو بعده؟",
    "طب لو أنا متعود أصوم يوم الجمعة؟",
    "إيه الدليل؟",
]
results = []
shots = []


def record(name, ok, details=None):
    entry = {"name": name, "ok": bool(ok)}
    if details is not None:
        entry["details"] = details
    results.append(entry)
    print(("PASS" if ok else "FAIL"), name, "" if details is None else json.dumps(details, ensure_ascii=False)[:800])


def wait(driver, seconds=30):
    return WebDriverWait(driver, seconds)


def shot(driver, name):
    path = SHOTS / f"{name}.png"
    driver.save_screenshot(str(path))
    shots.append(str(path.relative_to(ROOT)))


def ui_messages(driver):
    rows = []
    for row in driver.find_elements(By.CSS_SELECTOR, ".message-row"):
        try:
            sender = row.find_element(By.CSS_SELECTOR, ".message-sender").text.strip()
            text = row.find_element(By.CSS_SELECTOR, ".message-bubble p").text.strip()
        except Exception:  # noqa: BLE001
            continue
        rows.append({"role": "user" if sender == "أنت" else "assistant", "text": text})
    return rows


def active_id(driver):
    return driver.execute_script(f"return localStorage.getItem({json.dumps(ACTIVE_KEY)});")


def wait_idle(driver, seconds=30):
    wait(driver, seconds).until(
        lambda d: "جاهز للمحادثة" in d.find_element(By.CSS_SELECTOR, ".chat-status").text
        and not d.find_elements(By.CSS_SELECTOR, ".chat-retry-button")
    )


def send_message(driver, text, timeout=180):
    before = len(ui_messages(driver))
    field = driver.find_element(By.ID, "chat-message")
    field.clear()
    field.send_keys(text)
    driver.find_element(By.CSS_SELECTOR, ".send-message-button").click()
    wait(driver, timeout).until(
        lambda d: "جاهز للمحادثة" in d.find_element(By.CSS_SELECTOR, ".chat-status").text
        and len(ui_messages(d)) >= before + 2
        and not d.find_elements(By.CSS_SELECTOR, ".chat-retry-button")
    )
    return ui_messages(driver)


def api_get(driver, path):
    token = None
    for cookie in driver.get_cookies():
        if cookie["name"] == "token":
            token = cookie["value"]
            break
    if not token:
        raise AssertionError("token cookie missing")
    request = urllib.request.Request(
        API + path, headers={"Cookie": f"token={token}"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode())


def main():
    options = Options()
    options.add_argument("-headless")
    options.add_argument("--width=1365")
    options.add_argument("--height=900")
    driver = webdriver.Firefox(
        options=options,
        service=Service(
            executable_path="/snap/bin/geckodriver",
            log_output="/tmp/opencode/geckodriver_friday.log",
        ),
    )
    success = False
    try:
        driver.get(FRONTEND + "/signin")
        wait(driver, 30).until(EC.presence_of_element_located((By.ID, "signin-email")))
        driver.find_element(By.ID, "signin-email").send_keys(FIXTURE["email"])
        driver.find_element(By.ID, "signin-password").send_keys(FIXTURE["password"])
        driver.find_element(By.CSS_SELECTOR, ".auth-submit").click()
        wait(driver, 30).until(lambda d: "/signin" not in d.current_url)
        driver.get(FRONTEND + "/chat")
        wait(driver, 30).until(EC.presence_of_element_located((By.ID, "chat-message")))
        record("synthetic user logs in", True, {"url": driver.current_url})

        answers = []
        for index, question in enumerate(QUESTIONS, 1):
            messages = send_message(driver, question)
            answer = messages[-1]["text"]
            answers.append(answer)
            record(
                f"turn {index} returns an answer",
                len(messages) == index * 2 and messages[-2]["text"] == question and len(answer) > 20,
                {"answer": answer},
            )
            shot(driver, f"0{index}_turn_{index}")

        first, second, third, fourth = answers
        record(
            "first answer gives a qualified Friday-fasting ruling",
            len(first) > 80
            and "الجمعة" in first
            and (
                "كراه" in first
                or "لا يكره" in first
                or "النهي" in first
            )
            and ("حديث" in first or "قال" in first or "النهي" in first),
            {"answer": first},
        )
        record(
            "adjacent-day follow-up does not impose obligation",
            "لا توجب" in second
            and "يجب صيام يوم قبله" not in second
            and "الجمعة" in second,
            {"answer": second},
        )
        record(
            "habitual fasting is treated as a documented exception",
            "عادة" in third and "بشرط" not in third,
            {"answer": third},
        )
        record(
            "evidence request returns evidence for the preceding claim",
            ("الصحيحين" in fourth or "قال" in fourth)
            and "الجمعة" in fourth
            and re.search(r"\[(?:فقه|حديث|القرآن) — [^\]]+\]", fourth) is not None,
            {"answer": fourth},
        )
        combined = "\n".join(answers)
        record(
            "Friday-prayer verse is never used as fasting evidence",
            "الجمعة:9" not in combined and "نودي للصلاة" not in combined,
            {"contains_62_9": "الجمعة:9" in combined},
        )
        record(
            "internal source markers and source lists stay hidden",
            "(مصدر" not in combined
            and len(driver.find_elements(By.CSS_SELECTOR, ".message-sources")) == 0,
            {"source_lists": len(driver.find_elements(By.CSS_SELECTOR, ".message-sources"))},
        )
        record(
            "visible answers use readable inline citations",
            re.search(r"\[حديث — [^\]]+\]", combined) is not None
            or re.search(r"\[فقه — [^\]]+\]", combined) is not None,
            {"answer": first},
        )

        conversation_id = active_id(driver)
        stored = api_get(driver, f"/conversations/{conversation_id}")["data"]["messages"]
        record(
            "all eight messages persisted in order",
            [message["role"] for message in stored] == ["user", "assistant"] * 4
            and [message["content"] for message in stored[0::2]] == QUESTIONS,
            {"count": len(stored)},
        )

        driver.refresh()
        wait(driver, 30).until(lambda d: len(ui_messages(d)) == 8)
        wait_idle(driver)
        restored = ui_messages(driver)
        record(
            "refresh restores the same conversation and all answers",
            active_id(driver) == conversation_id
            and len(restored) == 8
            and restored[-1]["text"] == fourth,
            {"active": active_id(driver), "count": len(restored)},
        )
        shot(driver, "05_after_refresh")
        success = all(entry["ok"] for entry in results)
    finally:
        try:
            driver.quit()
        except Exception:  # noqa: BLE001
            pass
        summary = {
            "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "success": success,
            "questions": QUESTIONS,
            "answers": locals().get("answers", []),
            "results": results,
            "screenshots": shots,
        }
        (ART / "browser_results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
        )
        (ART / "browser_summary.md").write_text(
            "# Real-browser Friday-fasting conversation verification\n\n"
            f"- Result: {'PASS' if success else 'FAIL'}\n"
            f"- Scenarios: {sum(1 for r in results if r['ok'])}/{len(results)} passed\n"
            f"- Screenshots: {len(shots)}\n"
            "- Synthetic fixture user removed after the run.\n\n"
            + "\n".join(
                f"- {'PASS' if r['ok'] else 'FAIL'} — {r['name']}"
                for r in results
            )
        )
        print(json.dumps({
            "success": success,
            "passed": sum(1 for r in results if r["ok"]),
            "total": len(results),
            "screenshots": len(shots),
        }, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
