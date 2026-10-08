def spatiotemporal_interpolation(temporal, spatial, df, temporal_weight):
    df = df.copy()
    temporal_result = temporal(df.copy())
    spatial_result = spatial(df.copy())
    spatial_weight = 1 - temporal_weight

    result = df.copy()
    traffic_columns = ["speed_kmh", "flow_vph", "occupancy"]
    for col in traffic_columns:
            combined = (temporal_weight * temporal_result[col]) + (spatial_weight * spatial_result[col])
            combined = combined.fillna(temporal_result[col])
            combined = combined.fillna(spatial_result[col])
            result[col] = combined
    return result
