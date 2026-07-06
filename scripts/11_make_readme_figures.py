from pathlib import Path

import matplotlib.pyplot as plt

import CARSBench as cb


def main():
    output_dir = Path("docs/assets")
    output_dir.mkdir(parents=True, exist_ok=True)

    batch = cb.generate_dataset(
        num_samples=3,
        domain_name="A_typical",
        seed=42,
    )

    sample = batch.samples[0]

    plt.figure(figsize=(8, 4))
    plt.plot(sample.axis, sample.spectrum, label="Simulated CARS/BCARS spectrum")
    plt.plot(
        sample.axis, sample.raman_target, label="Raman-equivalent target", alpha=0.8
    )

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Signal")
    plt.title("Example CARSBench simulated spectrum")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "example_spectrum.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
