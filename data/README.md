# Data

All images come from the Galaxy Merger Challenge (Margalef-Bentabol et al. 2024) and are not redistributed here.

## TNG

The training stack `images_tng.npy` holds 463,071 IllustrisTNG mock HSC i-band stamps from the challenge's TNG training catalogue: the `train/` and `val/` folders of the four snapnum tarballs, minus catalogue rows with no image file. Each stamp is preprocessed as follows:

1. centre-cropped to a square and resampled to 128 x 128 (block mean when the native size is a multiple of 128, linear interpolation otherwise);
2. arcsinh-stretched, `arcsinh(x / 0.1) / arcsinh(1 / 0.1)`;
3. clipped to its own 0.5th-99.5th percentile range and rescaled to [0, 1].

This is the same `_resize_to_square` and `normalize_image` code as in `prepare_transfer_data.py`, which was copied from the TNG preprocessing so that Horizon-AGN and HSC images reach the encoder with the same pixel statistics.

`labels_tng.npy` is a structured array aligned row by row with the stack. Its fields include `ID`, `redshift`, `is_major_merger`, the stage flags `is_pre_merger` / `is_ongoing_merger` / `is_post_merger`, `sequence_ID` (merger tree), `data_type` (tng100 / tng300) and `subfolder`. `labels_tng_seqsplit.npy` has the same rows; its `subfolder` column defines the train/val rows used by the transfer section of `results.ipynb`.

The held-out set used for Tables 4.1–4.4 and the clustering is not read from either file. It is drawn in the notebooks with a `GroupShuffleSplit` on `sequence_ID` (seed 42, 20%), giving 93,347 galaxies with no merger tree shared with the training rows.

## Horizon-AGN and HSC

Built with `prepare_transfer_data.py --dataset horizon|hsc --extract --build`. Images are matched to catalogue rows by the integer ID at the end of each filename, keeping one projection per galaxy. The Horizon-AGN stack has 188,869 images (stages derived from the catalogue's `dt`), and the HSC stack has 123,818, of which 3,006 carry visual labels (`labeled == 1`).
