import subprocess
import sys

import pytest

TIMEOUT = 2

GC_WITH_IDLE_WORKERS = """
import gc, time
import spinningjenny

executor = spinningjenny.ThreadPoolExecutor(2)
time.sleep(0.2)
gc.collect()
print("ok")
"""

GC_IN_TASK_WHILE_CONSUMER_WAITS = """
import gc
import spinningjenny

executor = spinningjenny.ThreadPoolExecutor(1)
list(executor.map(lambda _: gc.collect(), [1]))
print("ok")
"""


@pytest.mark.parametrize(
    "code",
    [GC_WITH_IDLE_WORKERS, GC_IN_TASK_WHILE_CONSUMER_WAITS],
    ids=["gc-with-idle-workers", "gc-in-task-while-consumer-waits"],
)
def test_no_interpreter_deadlock(code):
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"deadlocked: subprocess did not exit within {TIMEOUT}s")
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
