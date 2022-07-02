## Models for Information Retrieval

Proyecto investigativo con el objetivo de encontrar un buen sistema para realizar selecciones de textos coherentes con un contexto dado dentro del inmenso corpus de los textos escrito por cubanos en las distintas redes sociales. En este repositorio se encuentra las distintas implementaciones realizadas a lo largo de la investigación las cuales se enumeran a continuación:

- `SRI con Modelo Vectorial`: modelo clásico del campo de la recuperación de información basado en el álgebra vectorial. [(link al código)](https://github.com/cuban-digital-language/Models-for-Information-Retrieval/blob/main/src/vectorial_model.py)
- `SRI con Modelo Probabilístico`: modelo clásico del campo de la recuperación de información basado en el teoría de las probabilidades, en este caso con la visión estimativa y frecuentista de las probabilidades. [(link al código)](https://github.com/cuban-digital-language/Models-for-Information-Retrieval/blob/main/src/probabilistic_model.py)
- `Thesaurus Strength`: Diccionario de sinónimos, con un enfoque de construcción automática, donde se definen la propiedad _"sinónimo de"_ por la correlación textual de los términos. [(link al código)](https://github.com/cuban-digital-language/Models-for-Information-Retrieval/blob/main/src/strength_thesaurus.py)
- `Thesaurus Bayesiano`: Diccionario de sinónimos, con un enfoque de construcción automática, donde se definen la propiedad _"sinónimo de"_ es inferida por una red de probabilidades bayesianas. [(link al código)](https://github.com/cuban-digital-language/Models-for-Information-Retrieval/blob/main/src/bayesian_thesaurus.py)

##### Lecturas Relacionadas:

- [Informe de la Investigación](https://github.com/cuban-digital-language/methodology/blob/main/sri.md)
- [Investigación derivada, para encontrar algún mecanismo automático para evaluar estos sistemas](https://github.com/cuban-digital-language/methodology/blob/main/cluster-dateset.md)

