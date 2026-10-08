def mask(df):
    df_copy = df.copy()
    not_missing = df_copy["speed_kmh"].notnull() & df_copy["flow_vph"].notnull()
    approved_indices = df_copy.index[not_missing]
    amount_to_mask = int(len(approved_indices) * 0.2)
    masked_indices = df_copy.loc[approved_indices].sample(n=amount_to_mask, random_state=42).index

    true_values = df_copy.loc[masked_indices, ["speed_kmh", "flow_vph"]].copy()
    df_copy.loc[masked_indices, ["speed_kmh", "flow_vph"]] = None

    return df_copy, true_values, masked_indices
