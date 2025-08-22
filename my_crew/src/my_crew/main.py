import sys
import warnings
from datetime import datetime as dt
import os

from my_crew import MyCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    topic = os.environ.get("TOPIC", "The future of AI in rural healthcare")
    os.makedirs("output", exist_ok=True)
    result = MyCrew().crew().kickoff(inputs={
        "topic": topic,
        "manager_brief": "",
        "run_timestamp": dt.now().isoformat(timespec="seconds"),  
    })
    print("\n=== FINAL REPORT (path: output/content_report.md) ===\n")
    print(result.raw)
