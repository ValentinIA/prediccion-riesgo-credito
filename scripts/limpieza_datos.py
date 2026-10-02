def limpieza_datos(df):
    # Copia explícita para no modificar el DataFrame original
    df_clean = df.copy()

    # 1. Renombrar columnas
    df_clean = df_clean.rename(columns={
        'person_age': 'edad',
        'person_income': 'ingreso',
        'person_home_ownership': 'propiedad_vivienda',
        'person_emp_length': 'duracion_empleo',
        'loan_intent': 'intencion_prestamo',
        'loan_grade': 'calificacion_prestamo',
        'loan_amnt': 'monto_prestamo',
        'loan_int_rate': 'tasa_interes',
        'loan_status': 'estado_prestamo',
        'loan_percent_income': 'porcentaje_ingreso',
        'cb_person_default_on_file': 'incumplimiento_historial',
        'cb_person_cred_hist_length': 'duracion_credito'
    })

    # 2. Eliminar duplicados
    df_clean = df_clean.drop_duplicates()

    # 3. Eliminar registros con nulos en tasa_interes
    df_clean = df_clean.dropna()

    # 4. Definir máscaras booleanas para MANTENER los datos válidos
    es_valida = (df_clean['duracion_empleo'] < 60) & (df_clean['edad'] < 100)

    # Aplicar todos los filtros en una sola operación
    df_clean = df_clean[es_valida]

    # 5. Eliminar columnas innecesarias
    cols_a_eliminar = [
        'calificacion_prestamo',
        'incumplimiento_historial',
        'edad',
        'monto_prestamo'
    ]
    
    df_clean = df_clean.drop(columns=cols_a_eliminar, errors='ignore')

    return df_clean