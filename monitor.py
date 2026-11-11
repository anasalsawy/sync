"""Watch Sync runtime events locally: python monitor.py PATH."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--run", help="Show events for this run label only")
    args = parser.parse_args()
    print(f"Watching {args.path}", flush=True)
    try:
        while not args.path.exists():
            time.sleep(0.25)
        with args.path.open(encoding="utf-8") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line.endswith("\n"):
                    stream.seek(offset)
                    time.sleep(0.25)
                    continue
                record = json.loads(line)
                timestamp = datetime.fromtimestamp(record.pop("time")).strftime("%H:%M:%S")
                run = record.pop("run")
                if args.run and run != args.run:
                    continue
                event = record.pop("event")
                print(f"{timestamp}  {run}  {event}  {json.dumps(record, ensure_ascii=False)}",
                      flush=True)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
