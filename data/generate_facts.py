from argparse import ArgumentParser

from datasets import Dataset

from .dataset import fact_generator

if __name__ == "__main__":
    from argparse import ArgumentParser

    from datasets import Dataset

    parser = ArgumentParser()
    parser.add_argument("--num_facts", type=int, default=1000)
    args = parser.parse_args()

    dataset = fact_generator(args.num_facts)
    Dataset.from_list(list(dataset)).save_to_disk("data/facts_dataset.hf")
