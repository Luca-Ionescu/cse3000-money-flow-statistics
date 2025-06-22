import pandas as pd
import spartan as st

def run_cubeflow(k: int):
    offset = 0
    id_to_int_per_layer = []
    int_to_id_per_layer = []

    file_list = [f"./output-cubeflow/fs{i}.csv" for i in range(1, k+1)]
    df_combined = pd.concat((pd.read_csv(f, header=0, names=["idx", "src", "dst", "time", "amount"]) for f in file_list), ignore_index = True)
    unique_ids = pd.Index(df_combined["src"]).append(pd.Index(df_combined["dst"])).unique()

    id_to_int = {acc_id: i for i, acc_id in enumerate(unique_ids)}
    int_to_id = {i: acc_id for acc_id, i in id_to_int.items()}

    for i in range(1, k+1):
        df = pd.read_csv("./output-cubeflow/fs" + str(i) + ".csv", index_col=0)
        df["0"] = df["0"].map(id_to_int)
        df["1"] = df["1"].map(id_to_int)
        df.to_csv(f"./encoded-output-cubeflow/fs{i}.csv", columns=["0", "1", "2", "3"], index=False,
                  header=False)

    # for i in range(1, k+1):
    #     int_to_id = {}
    #     df = pd.read_csv(f"./output-cubeflow/fs{i}.csv", header=0, names = ["idx", "src", "dst", "timestamp", "amount"])
    #
    #     if i == 1:
    #         unique_ids_src = pd.Index(df["src"]).unique()
    #         id_to_int = {acc_id: idx + offset for idx, acc_id in enumerate(unique_ids_src)}
    #         offset += len(unique_ids_src)
    #         id_to_int_per_layer.append(id_to_int)
    #         for acc_id, idx in id_to_int.items():
    #             int_to_id[idx] = acc_id
    #         int_to_id_per_layer.append(int_to_id)
    #
    #         unique_ids_dst = pd.Index(df["dst"]).unique()
    #         id_to_int = {acc_id: idx + offset for idx, acc_id in enumerate(unique_ids_dst)}
    #         offset += len(unique_ids_dst)
    #         id_to_int_per_layer.append(id_to_int)
    #         int_to_id = {}
    #         for acc_id, idx in id_to_int.items():
    #             int_to_id[idx] = acc_id
    #         int_to_id_per_layer.append(int_to_id)
    #
    #     else:
    #         unique_ids = pd.Index(df["dst"]).unique()
    #
    #         # Map for this file: account ID -> integer starting from offset
    #         id_to_int = {acc_id: idx + offset for idx, acc_id in enumerate(unique_ids)}
    #
    #         # Update global int_to_id with new mappings
    #         for acc_id, idx in id_to_int.items():
    #             int_to_id[idx] = acc_id
    #
    #         id_to_int_per_layer.append(id_to_int)
    #         int_to_id_per_layer.append(int_to_id)
    #         offset += len(unique_ids)
    #
    #     df["src"] = df["src"].map(id_to_int_per_layer[i - 1])
    #     df["dst"] = df["dst"].map(id_to_int_per_layer[i])



    #balance_cubeflow_data("./encoded-output-cubeflow/fs1.csv", "./encoded-output-cubeflow/fs2.csv")

    tensor_data = []

    # balance_cubeflow_data("./encoded-output-cubeflow/fs1.csv", "./encoded-output-cubeflow/fs2.csv")

    for i in range(1, k+1):
        tensor_data.append(st.loadTensor(path= "./encoded-output-cubeflow/fs" + str(i) + ".csv", header=0))

    tensor_array = []
    for tens in tensor_data:
        stensor = tens.toSTensor(hasvalue=True)

        tensor_array.append(stensor)

    cf = st.CubeFlow(tensor_array, alpha=0.2, k=9, dim=3, outpath='')
    res = cf.run(del_type=1, maxsize=-1)
    print(res)

    return res, int_to_id
