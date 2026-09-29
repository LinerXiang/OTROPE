"""Merge judge shards in shard order, without an unseeded shuffle."""
import argparse
import json
from pathlib import Path


def merge_shards(base_path, output_path, num_shards):
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    records = []
    for index in range(num_shards):
        path = Path(f"{base_path}{index}.json")
        with path.open(encoding="utf-8") as handle:
            shard = json.load(handle)
        if not isinstance(shard, list):
            raise ValueError(f"Expected a JSON list in {path}")
        records.extend(shard)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--output_dir", required=True, help="Output JSON file")
    parser.add_argument("--num_datasets", type=int, required=True)
    args = parser.parse_args()
    print(f"Merged {merge_shards(args.base_path, args.output_dir, args.num_datasets)} records")


if __name__ == "__main__":
    main()
