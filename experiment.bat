python .\defeasible_intervention.py  --task snli --model mnli_deberta --intervention weakener
python .\defeasible_intervention.py  --task snli --model mnli_deberta
python .\defeasible_intervention.py  --task atomic --model mnli_deberta --intervention weakener
python .\defeasible_intervention.py  --task atomic --model mnli_deberta
python .\defeasible_intervention.py  --task social_chem --model mnli_deberta --intervention weakener
python .\defeasible_intervention.py  --task social_chem --model mnli_deberta

python .\defeasible_intervention.py  --task snli --model anli_roberta --intervention weakener
python .\defeasible_intervention.py  --task snli --model anli_roberta
python .\defeasible_intervention.py  --task atomic --model anli_roberta --intervention weakener
python .\defeasible_intervention.py  --task atomic --model anli_roberta
python .\defeasible_intervention.py  --task social_chem --model anli_roberta --intervention weakener
python .\defeasible_intervention.py  --task social_chem --model anli_roberta

@REM python .\defeasible_intervention.py  --task snli --model anli_xlnet --intervention weakener
@REM python .\defeasible_intervention.py  --task snli --model anli_xlnet
@REM python .\defeasible_intervention.py  --task atomic --model anli_xlnet --intervention weakener
@REM python .\defeasible_intervention.py  --task atomic --model anli_xlnet
@REM python .\defeasible_intervention.py  --task social_chem --model anli_xlnet --intervention weakener
@REM python .\defeasible_intervention.py  --task social_chem --model anli_xlnet