import pandas as pd
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Sample a subset of users to control LLM costs.")
    parser.add_argument("--profiles_in", required=True, help="Input profiles_k*.csv")
    parser.add_argument("--interactions_in", required=True, help="Input interactions_k*.csv")
    parser.add_argument("--n_users", type=int, default=1000, help="Number of users to sample")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    df_prof = pd.read_csv(args.profiles_in)
    df_inter = pd.read_csv(args.interactions_in)
    
    # 1. Randomly sample users
    all_users = df_inter['user_id'].unique()
    np.random.seed(42)  # For reproducibility
    sampled_users = np.random.choice(all_users, size=min(args.n_users, len(all_users)), replace=False)
    
    # 2. Filter interactions to only contain sampled users
    filtered_inter = df_inter[df_inter['user_id'].isin(sampled_users)]
    
    # 3. Get items connected to the sampled users
    connected_items = filtered_inter['item_id'].unique()
    
    # 4. Filter profiles that only correspond to valid user-item pairs
    valid_pairs = set(zip(filtered_inter['user_id'].astype(str), filtered_inter['item_id'].astype(str)))
    
    df_prof['user_id_str'] = df_prof['user_id'].astype(str)
    df_prof['item_id_str'] = df_prof['item_id'].astype(str)
    df_prof['pair'] = list(zip(df_prof['user_id_str'], df_prof['item_id_str']))
    
    filtered_prof = df_prof[df_prof['pair'].isin(valid_pairs)].drop(columns=['pair', 'user_id_str', 'item_id_str'])
    
    prof_out = os.path.join(args.out_dir, "profiles.csv")
    inter_out = os.path.join(args.out_dir, "interactions.csv")
    
    filtered_prof.to_csv(prof_out, index=False)
    filtered_inter.to_csv(inter_out, index=False)
    
    print(f"[Sample Success] Sampled {len(sampled_users)} users and connected {len(connected_items)} items.")
    print(f"Resulting interactions: {len(filtered_inter)} rows.")
    print(f"Saved: {prof_out} and {inter_out}")

if __name__ == "__main__":
    main()
