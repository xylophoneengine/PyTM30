# Data tables: licence and attribution

The spectral data tables in this directory are **not** covered by the MIT
licence of the PyTM30 code. They are CIE datasets, or adaptations of CIE
datasets, and are distributed under the
[Creative Commons Attribution-ShareAlike 4.0 International licence](https://creativecommons.org/licenses/by-sa/4.0/)
(CC BY-SA 4.0), the licence under which the CIE publishes them.

Creator of all source datasets: International Commission on Illumination
(CIE), Vienna, Austria. Each source dataset is published at
<https://cie.co.at/data-tables> under the DOI listed below.

The tables were extracted with colour-science 0.4.7 (BSD-3-Clause), used as
a tool; see `tools/generate_data_colour_science.py` for the exact calls. The
numbers themselves are the CIE's. `planckian_uv.csv` is computed by
`tools/generate_planckian_lut.py`.

| File | Source dataset (CIE, CC BY-SA 4.0) | Changes made here |
|---|---|---|
| `cmf_1931_2.csv` | CIE 2019, Colour-matching functions of CIE 1931 standard colorimetric observer, [10.25039/CIE.DS.xvudnb9b](https://doi.org/10.25039/CIE.DS.xvudnb9b) | none (360-830 nm, 1 nm) |
| `cie_1931_2.csv` | as above | trimmed to 380-780 nm |
| `cmf_1964_10.csv` | CIE 2019, Colour-matching functions of CIE 1964 standard colorimetric observer, [10.25039/CIE.DS.sqksu2n5](https://doi.org/10.25039/CIE.DS.sqksu2n5) | z-bar entries the CIE leaves empty (560-830 nm) written as 0 or as float noise of order 1e-20 |
| `cmf_2006_2.csv`, `cmf_2015_2.csv` | CIE 2015, Cone-fundamental-based spectral tristimulus values, 2 deg (CIE 170-2:2015, Table 10.7a), [10.25039/CIE.DS.548rw69q](https://doi.org/10.25039/CIE.DS.548rw69q) | zero-padded 360-389 nm; the same table is stored under both names |
| `cmf_2006_10.csv`, `cmf_2015_10.csv` | CIE 2015, Cone-fundamental-based spectral tristimulus values, 10 deg (CIE 170-2:2015, Table 10.8), [10.25039/CIE.DS.dm6qiig7](https://doi.org/10.25039/CIE.DS.dm6qiig7) | zero-padded 360-389 nm; the same table is stored under both names |
| `daylight_basis.csv` | CIE 2018, Components of relative spectral distribution of daylight (CIE 015:2018, Table 6), [10.25039/CIE.DS.w7zunnny](https://doi.org/10.25039/CIE.DS.w7zunnny) | trimmed to 380-780 nm |
| `d65_1nm.csv` | computed from `daylight_basis.csv` by the CIE D-series method; compare CIE D65, [10.25039/CIE.DS.hjfjmt59](https://doi.org/10.25039/CIE.DS.hjfjmt59) | recomputed with linear interpolation of the basis (agrees with the CIE table to 9e-4) |
| `illuminant_a_1nm.csv` | CIE standard illuminant A, [10.25039/CIE.DS.8jsxjrsn](https://doi.org/10.25039/CIE.DS.8jsxjrsn) | evaluated from the ISO/CIE 11664-2 formula (agrees with the CIE table to 5e-4) |
| `fl1_1nm.csv` ... `fl12_1nm.csv` | CIE 2018, Relative spectral power distributions of illuminants representing typical fluorescent lamps, 5 nm (CIE 015:2018, Tables 10.1-10.3), [10.25039/CIE.DS.ukaymjdn](https://doi.org/10.25039/CIE.DS.ukaymjdn) | Sprague-interpolated from 5 nm to 1 nm. This is **not** the CIE's own 1 nm table ([10.25039/CIE.DS.54hy6srn](https://doi.org/10.25039/CIE.DS.54hy6srn)), which resolves the emission lines and differs substantially |
| `hp1_5nm.csv` ... `hp5_5nm.csv` | CIE 2018, High-pressure lamps HP1-HP5 (CIE 015:2018, Table 11), [10.25039/CIE.DS.f6rvvnev](https://doi.org/10.25039/CIE.DS.f6rvvnev) | none |
| `ces.csv` | CIE 2017, CIE 2017 colour fidelity index test samples, 1 nm (CIE 224:2017), [10.25039/CIE.DS.8svs5rqd](https://doi.org/10.25039/CIE.DS.8svs5rqd) | none |
| `ces_5nm.csv` | CIE 2017, CIE 2017 colour fidelity index test samples, 5 nm (CIE 224:2017, Tables A.1-A.10), [10.25039/CIE.DS.wi5idbqu](https://doi.org/10.25039/CIE.DS.wi5idbqu) | none |
| `planckian_uv.csv` | computed here from Planck's law and the CIE 1931 CMFs ([10.25039/CIE.DS.xvudnb9b](https://doi.org/10.25039/CIE.DS.xvudnb9b)) | own computation; 1489 CCTs, 1000-41073 K |

`illuminant_corpus.txt` is a list of file names written for this project
and is covered by the MIT licence.

Further credit, not required by any licence: the 99 colour evaluation
samples originate with ANSI/IES TM-30 (IES), and the CIE 2006/2015
physiologically based CMFs derive from Stockman & Sharpe (2000), as
distributed by the Colour & Vision Research Laboratory (CVRL,
<http://www.cvrl.org>).

The CC BY-SA 4.0 licence comes with no warranty; see its disclaimer
(Section 5).
