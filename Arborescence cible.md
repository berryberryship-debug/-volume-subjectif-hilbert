├── pipeline_principal/
│   ├── local_jnd_bridge_calibration.py
│   ├── volume_subjectif_formats.py
│   ├── pipeline_volume_subjectif_complet.py
│   ├── conjunctive_validation_and_power.py
│   ├── quality_controls.py
│   └── test_synthetic_alignment.py
│
├── exploratory/
│   ├── vacuite_holotelegraphique_unifie.py
│   ├── v95_skyrme_soliton.py
│   └── README.md          # ces modules ne sont pas importés par pipeline_principal/
│
├── PROTOCOL.md
├── REGISTRE_REVENDICATIONS.md
└── README.md




! grep -rn "from exploratory" pipeline_principal/
! grep -rn "import exploratory" pipeline_principal/
