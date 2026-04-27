"""Wrapper: rerun PhySO PCA SR to a fresh output directory."""
import os, sys

def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cosmo_symreg_physo as base
    base.OUTPUT_DIR        = os.path.join(os.path.dirname(__file__), "outputs_rerun")
    base.MAX_N_EVALUATIONS = int(2e5)
    base.N_EPOCHS          = int(10)
    base.N_COMPONENTS      = 3
    os.makedirs(base.OUTPUT_DIR, exist_ok=True)

    cosmo, dv, x_arr = base.load_data(base.CSV_PATH)
    base.run_pca_sr(cosmo, dv, x_arr, n_components=base.N_COMPONENTS)
    print("\n✓ PCA rerun done →", base.OUTPUT_DIR)


if __name__ == "__main__":
    main()
