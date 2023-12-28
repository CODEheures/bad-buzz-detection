from streamlit_extras.switch_page_button import switch_page
from experiment import pages_management
import streamlit as st


def main():
    """Entry point for streamlit Train App
    """
    pages_management.update_pages()
    switch_page("Accueil")


def test_function():
    """A faker function to run pytest on github pipeline

    Returns:
        int: a random value to test in pytest
    """
    st.write('Hello world Sylvain')
    return 5


if __name__ == "__main__":
    main()
