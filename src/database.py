
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import Config


def get_engine() -> Engine:
    """
    Create and return a SQLAlchemy database engine.
    """
    return create_engine(Config.database_url())


def test_connection() -> bool:
    """
    Test the PostgreSQL database connection.
    """
    engine = get_engine()

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 AS connection_test;"))
            value = result.scalar()
            return value == 1
    except Exception as error:
        print("Database connection failed:")
        print(error)
        return False


def fetch_companies():
    """
    Fetch company records from the companies table.
    """
    engine = get_engine()

    query = text("""
        SELECT company_id, company_name, ticker, industry
        FROM companies
        ORDER BY company_id;
    """)

    with engine.connect() as conn:
        result = conn.execute(query)
        return result.fetchall()


if __name__ == "__main__":
    is_connected = test_connection()

    if is_connected:
        print("Database connection successful.")

        companies = fetch_companies()
        print("Companies:")
        for company in companies:
            print(company)
    else:
        print("Database connection failed.")