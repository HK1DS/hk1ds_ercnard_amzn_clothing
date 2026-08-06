import pandas as pd
import argparse
import os
import numpy as np


def build_cover_profiles(df_prof, sampled_users, connected_items):
    """Keep one profile per sampled entity and rebuild a compact cover.

    profiles.csv is allowed to be an entity cover rather than a table of real
    interactions (see apply_k_core.py).  Filtering it by observed user-item
    pairs therefore drops valid profile text.  Stage B only needs each selected
    user and item to occur at least once, so pair the two entity lists again.
    """
    prof = df_prof.copy()
    prof["user_id"] = prof["user_id"].astype(str)
    prof["item_id"] = prof["item_id"].astype(str)
    users = (
        prof[prof["user_id"].isin(sampled_users)]
        .drop_duplicates("user_id")
        .set_index("user_id")["user_profile"]
        .to_dict()
    )
    items = (
        prof[prof["item_id"].isin(connected_items)]
        .drop_duplicates("item_id")
        .set_index("item_id")["item_profile"]
        .to_dict()
    )
    missing_users = sorted(set(sampled_users) - set(users))
    missing_items = sorted(set(connected_items) - set(items))
    if missing_users or missing_items:
        raise RuntimeError(
            "Missing profile text for sampled entities: "
            f"{len(missing_users)} users, {len(missing_items)} items"
        )
    user_ids, item_ids = sorted(users), sorted(items)
    n_rows = max(len(user_ids), len(item_ids))
    return pd.DataFrame([
        {
            "user_id": user_ids[i % len(user_ids)],
            "user_profile": users[user_ids[i % len(user_ids)]],
            "item_id": item_ids[i % len(item_ids)],
            "item_profile": items[item_ids[i % len(item_ids)]],
        }
        for i in range(n_rows)
    ])

def main():
    parser = argparse.ArgumentParser(description="Sample a subset of users to control LLM costs.")
    parser.add_argument("--profiles_in", required=True, help="Input profiles_k*.csv")
    parser.add_argument("--interactions_in", required=True, help="Input interactions_k*.csv")
    parser.add_argument("--n_users", type=int, default=1000, help="Number of users to sample")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    df_prof = pd.read_csv(args.profiles_in, dtype=str, keep_default_na=False)
    df_inter = pd.read_csv(
        args.interactions_in, dtype={"user_id": str, "item_id": str}
    )
    
    # 1. Randomly sample users
    all_users = df_inter['user_id'].unique()
    np.random.seed(42)  # For reproducibility
    sampled_users = np.random.choice(all_users, size=min(args.n_users, len(all_users)), replace=False)
    
    # 2. Filter interactions to only contain sampled users
    filtered_inter = df_inter[df_inter['user_id'].isin(sampled_users)]
    
    # 3. Get items connected to the sampled users
    connected_items = filtered_inter['item_id'].unique()
    
    # 4. Rebuild an entity cover. Profiles rows are not necessarily observed
    # user-item interactions, especially after apply_k_core.py compaction.
    filtered_prof = build_cover_profiles(
        df_prof,
        set(map(str, sampled_users)),
        set(map(str, connected_items)),
    )
    
    prof_out = os.path.join(args.out_dir, "profiles.csv")
    inter_out = os.path.join(args.out_dir, "interactions.csv")
    
    filtered_prof.to_csv(prof_out, index=False)
    filtered_inter.to_csv(inter_out, index=False)
    
    print(f"[Sample Success] Sampled {len(sampled_users)} users and connected {len(connected_items)} items.")
    print(f"Resulting interactions: {len(filtered_inter)} rows.")
    print(f"Saved: {prof_out} and {inter_out}")

if __name__ == "__main__":
    main()
