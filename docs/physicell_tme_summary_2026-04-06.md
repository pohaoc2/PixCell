# PhysiCell TME Summary

This note captures the earlier summary of how to turn [TME_model_plan.md](/home/ec2-user/PixCell/TME_model_plan.md#L1) into a runnable PhysiCell project, plus what PhysiCell can and cannot represent for different cancer types and clinical stages.

## 1. What the local file is

[TME_model_plan.md](/home/ec2-user/PixCell/TME_model_plan.md#L1) is a modeling plan, not a runnable PhysiCell project yet.

It describes:

- cancer, immune, and stromal cells
- oxygen and glucose fields
- vasculature
- proliferative, quiescent, and dead cell states
- a phased implementation plan

What is missing from this repo right now is the usual PhysiCell project structure such as:

- `Makefile`
- `main.cpp`
- `config/PhysiCell_settings.xml`
- `custom_modules/`

## 2. How to run PhysiCell on Ubuntu

On Ubuntu, first install the C/C++ build toolchain:

```bash
sudo apt update
sudo apt install -y build-essential
```

Then, in a local PhysiCell checkout:

```bash
cd /path/to/PhysiCell
make clean
make template
make
./project ./config/PhysiCell_settings.xml
```

For a starting point closer to this TME plan, use:

```bash
cd /path/to/PhysiCell
make clean
make cancer-metabolism-sample
make
./project ./config/PhysiCell_settings.xml
```

Notes:

- On a fresh PhysiCell clone, do not run `make reset` first. It can fail because `config/PhysiCell_settings-backup.xml` may not exist yet.
- If `make` fails with `g++: No such file or directory`, the build toolchain is missing or the shell needs to be refreshed after installation.
- `./project` only exists after a successful compile.

## 3. Recommended implementation path for this plan

The cleanest progression is:

1. Start from `template` or the workshop oxygen-tumor example.
2. Add oxygen-driven tumor growth first.
3. Add glucose and metabolism behavior.
4. Add immune cells and immune-kill rules.
5. Add static vasculature sources.
6. Add angiogenesis later if needed.

Where the biology usually goes:

- `config/PhysiCell_settings.xml`: domain, substrates, cell definitions, outputs
- `custom_modules/custom.cpp`: custom energy proxy and state-transition logic

For this plan, the energy proxy

```text
E = w_O * O + w_G * G
```

is most naturally implemented in custom code.

## 4. Can PhysiCell model different cancer types?

Yes.

PhysiCell is a general multicellular simulation framework rather than a model tied to one tumor type.

Examples and implications:

- breast cancer: yes; the original PhysiCell paper includes a ductal carcinoma in situ example
- colon or colorectal cancer: yes; official training materials explicitly point to colorectal carcinoma as a natural use case
- tumor-immune settings: yes; official sample and companion projects cover cancer-immune interactions

In practice, changing cancer type means changing:

- cell definitions
- growth and death parameters
- substrate usage
- motility and adhesion rules
- immune interactions
- vasculature and microenvironment assumptions

## 5. Can PhysiCell model Stage I, II, III breast cancer?

Yes in principle, but not as a built-in one-click option.

PhysiCell does not ship with a native concept like:

- Stage I breast cancer
- Stage II breast cancer
- Stage III breast cancer

Instead, stage is something you represent indirectly by calibrating different model variants or parameter sets for:

- tumor size and burden
- hypoxia and necrosis
- invasion and local spread
- angiogenesis or vascular remodeling
- immune infiltration or suppression
- proliferation and death balance

So the right interpretation is:

- different cancer types: supported
- different clinical stages: possible, but requires custom calibration and biological assumptions

## 6. Practical recommendation

For this TME plan, the most practical route is:

1. Run `cancer-metabolism-sample` first.
2. Inspect `cancer-immune-sample` next for immune behavior patterns.
3. Merge the pieces into a custom project for this TME plan.
4. Start with static vessels before attempting angiogenesis.

## 7. Sources

- PhysiCell paper: <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005991>
- Official site: <https://physicell.org/>
- Official GitHub repo: <https://github.com/MathCancer/PhysiCell>
- Official releases: <https://github.com/mathcancer/physicell/releases>
- Official workshop agenda: <https://github.com/physicell-training/ws2023/blob/main/agenda.md>
- PhysiCell Studio guide: <https://github.com/PhysiCell-Tools/Studio-Guide>
- Tumor-immune example: <https://github.com/MathCancer/pc4cancerimmune>
- Invasion example: <https://github.com/PhysiCell-Models/collective-invasion>
- Microvessel Chaste documentation: <https://jmsgrogan.github.io/MicrovesselChaste/documentation/>
