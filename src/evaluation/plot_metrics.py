from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_pe_comparison(save_dir, base_exp_dir, gen_plots=True, gen_tables=True):
    pe_csv = base_exp_dir / "pe_adversarial" / "GAN_Evasion_multimetric.csv"
    if not pe_csv.exists():
        print(f"[Warning] PE multi-metric file not found at {pe_csv}")
        return

    df = pd.read_csv(pe_csv)
    # 0.0x represents clean baseline performance
    df_clean = df[df['Scale'] == 0.0].copy()

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']

    if gen_tables:
        summary_df = df_clean[['Model'] + [m for m in metrics if m in df_clean.columns]]
        table_path = save_dir / "PE_General_Classification_Table.csv"
        summary_df.to_csv(table_path, index=False)
        print(f"[Success] Saved PE Comparison Table to {table_path}")

    if gen_plots:
        df_melt = df_clean.melt(
            id_vars='Model',
            value_vars=[m for m in metrics if m in df_clean.columns],
            var_name='Metric',
            value_name='Score'
        )

        plt.figure(figsize=(14, 6))
        sns.barplot(data=df_melt, x='Model', y='Score', hue='Metric', palette='viridis')
        plt.title("PE Domain: General Baseline Performance (Clean Data)", fontsize=14, fontweight='bold', pad=15)
        plt.ylim(0, 1.05)
        plt.ylabel("Score", fontsize=12)
        plt.xticks(rotation=30)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()

        plot_path = save_dir / "PE_General_Classification_Plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Success] Saved PE Comparison Plot to {plot_path}")


def generate_nids_comparison(save_dir, base_exp_dir, gen_plots=True, gen_tables=True):
    nids_csv = base_exp_dir / "nids_adversarial" / "Packet_Size_multimetric.csv"
    if not nids_csv.exists():
        print(f"[Warning] NIDS multi-metric file not found at {nids_csv}")
        return

    df = pd.read_csv(nids_csv)
    # 1.0x represents clean baseline performance
    df_clean = df[(df['Scale'] == '1.0x') | (df['Scale'] == 1.0)].copy()

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']

    if gen_tables:
        summary_df = df_clean[['Model'] + [m for m in metrics if m in df_clean.columns]]
        table_path = save_dir / "NIDS_General_Classification_Table.csv"
        summary_df.to_csv(table_path, index=False)
        print(f"[Success] Saved NIDS Comparison Table to {table_path}")

    if gen_plots:
        df_melt = df_clean.melt(
            id_vars='Model',
            value_vars=[m for m in metrics if m in df_clean.columns],
            var_name='Metric',
            value_name='Score'
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_melt, x='Model', y='Score', hue='Metric', palette='magma')
        plt.title("NIDS Domain: General Baseline Performance (Clean Data)", fontsize=14, fontweight='bold', pad=15)
        plt.ylim(0, 1.05)
        plt.ylabel("Score", fontsize=12)
        plt.xticks(rotation=20)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()

        plot_path = save_dir / "NIDS_General_Classification_Plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Success] Saved NIDS Comparison Plot to {plot_path}")