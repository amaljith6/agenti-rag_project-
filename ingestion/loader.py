from unstructured.partition.pdf import partition_pdf


def load_pdf(path: str):
    """
    Loads a PDF and returns unstructured elements.
    """
    elements = partition_pdf(
        filename=path,
        strategy="fast",
        infer_table_structure=False,
    )
    return elements
