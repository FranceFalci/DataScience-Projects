<!-- python -m venv .venv  creo el entorno virtual USAR VERSION ESTABLE POR EJEMPLO 3.11 -->
<!-- source .venv/Scripts/activate lo activo -->
<!-- pip freeze > requirements.txt -->
<!-- agregar dataset a dvc 
dvc add datasets/nuevas_radiografias -->
<!-- dvc push -->
<!-- instalo lo requerido  y pip freeze-->

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml