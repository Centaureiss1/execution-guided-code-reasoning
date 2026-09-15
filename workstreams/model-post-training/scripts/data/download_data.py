from datasets import load_dataset

ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train",
                  cache_dir="/home/brui/cs639_final/data/raw")
print(ds.column_names)
print(ds[0])
