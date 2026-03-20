import os
import pandas as pd

NUMBER_OF_BENIGN_SET = 2
seen_attacks = []


def check_seen_label(label):
    for i in range(len(seen_attacks)):
        if seen_attacks[i] in label:
            return True, i
    seen_attacks.append(label)
    return False, -1

def generate_datasets():
    DATASET_DIRECTORY = "datasets/"
    df_sets = [k for k in os.listdir(DATASET_DIRECTORY) if k.endswith(".csv")]
    df_sets.sort()

    # Begnin class
    df_mul = pd.read_csv(DATASET_DIRECTORY + df_sets[0])
    df_mul["Type"] = "Benign"

    df_binary = pd.read_csv(DATASET_DIRECTORY + df_sets[0])
    df_binary["Type"] = "Benign"

    for i in range(1, NUMBER_OF_BENIGN_SET):
        df = pd.read_csv(DATASET_DIRECTORY + df_sets[i])
        df["Type"] = "Benign"
        df_mul = pd.concat([df_mul, df])
        df_binary = pd.concat([df_binary, df])


    for i in range(2, len(df_sets)):
        df = pd.read_csv(DATASET_DIRECTORY + df_sets[i])
        percent_of_data = int(df.shape[0] * 1.00)
        # Label column for multiclasses classification
        label = df_sets[i].removeprefix("DDoS-").removesuffix(".pcap.csv")
        seen, idx = check_seen_label(label)
        # we already saw the type of attack so we use the correct label instead of XX-1 or any number
        # depending of the number of csv file of a certain type of attack
        # Eg : ACK_Fragmentation1 will be labeled ACK_Fragmentation
        if seen:
            df["Type"] = seen_attacks[idx]
        else:
            df["Type"] = label
        df_mul = pd.concat([df_mul, df.head(percent_of_data)])

        # Label colum for binary classification
        df["Type"] = "DDoS"
        df_binary = pd.concat([df_binary, df.head(percent_of_data)])

    df_mul.to_csv("dataset_multiclass.csv", index=False)
    df_binary.to_csv("dataset_binaire.csv", index=False)


    print("Check datasets")
    print(
        "Number of entries : ",
        df_binary.shape[0],
        " and ",
        df_mul.shape[0],
        " are equal ",
        df_binary.shape == df_mul.shape,
    )

    
if __name__ == "__main__":
    generate_datasets()