g++ -O3 -std=c++17 -shared -fPIC biophysics/pp1_i1.cpp -o biophysics/libpp1_i1.so
python3 conjunctive_validation_and_power.py
python3 pipeline_volume_subjectif_complet.py
