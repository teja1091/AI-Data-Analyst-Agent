
from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd


class DatasetLoader:
    """
    Load supported dataset files into Pandas DataFrames.

    Supported formats:
    - CSV (.csv)
    - Excel (.xlsx)

    The loader validates file input and returns a DataFrame.
    It does not modify the source file.
    """

    SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}

    @classmethod
    def load(
        cls,
        filename: str,
        file_content: bytes
    ) -> pd.DataFrame:

        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("A valid filename is required.")

        if not isinstance(file_content, bytes):
            raise TypeError("File content must be bytes.")

        if not file_content:
            raise ValueError("Uploaded file is empty.")

        extension = Path(filename.strip()).suffix.lower()

        if extension not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                "Unsupported file format. Supported formats: CSV and XLSX."
            )

        if extension == ".csv":
            df = cls._load_csv(file_content)
        else:
            df = cls._load_excel(file_content)

        return cls._validate_dataframe(df)

    @staticmethod
    def _load_csv(file_content: bytes) -> pd.DataFrame:
        """
        Parse CSV content using UTF-8 BOM or CP1252 fallback.
        """

        for encoding in ("utf-8-sig", "cp1252"):
            try:
                return pd.read_csv(
                    BytesIO(file_content),
                    encoding=encoding,
                    on_bad_lines="error"
                )

            except UnicodeDecodeError:
                continue

            except pd.errors.EmptyDataError as exc:
                raise ValueError(
                    "CSV file does not contain any data."
                ) from exc

            except pd.errors.ParserError as exc:
                raise ValueError(
                    "CSV file has an invalid structure."
                ) from exc

            except Exception as exc:
                raise ValueError(
                    "Unable to read the CSV file."
                ) from exc

        raise ValueError(
            "Unable to decode the CSV file using supported encodings."
        )

    @staticmethod
    def _load_excel(file_content: bytes) -> pd.DataFrame:
        """
        Validate and parse the first worksheet of an XLSX workbook.
        """

        buffer = BytesIO(file_content)

        try:
            with zipfile.ZipFile(buffer) as workbook:

                corrupted_file = workbook.testzip()

                if corrupted_file is not None:
                    raise ValueError(
                        "Excel workbook contains a corrupted file."
                    )

                required_files = {
                    "[Content_Types].xml",
                    "xl/workbook.xml"
                }

                if not required_files.issubset(workbook.namelist()):
                    raise ValueError(
                        "File is not a valid XLSX workbook."
                    )

            buffer.seek(0)

            return pd.read_excel(
                buffer,
                engine="openpyxl"
            )

        except ImportError as exc:
            raise RuntimeError(
                "Excel support requires the openpyxl package."
            ) from exc

        except zipfile.BadZipFile as exc:
            raise ValueError(
                "Excel file is corrupted or is not a valid XLSX archive."
            ) from exc

        except ValueError:
            raise

        except Exception as exc:
            raise ValueError(
                "Unable to read the Excel file. "
                "Ensure it is a valid XLSX workbook."
            ) from exc

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the resulting DataFrame.
        """

        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                "File could not be converted into a dataset."
            )

        if len(df.columns) == 0:
            raise ValueError(
                "Dataset does not contain any columns."
            )

        if df.empty:
            raise ValueError(
                "Dataset does not contain any data rows."
            )

        return df
