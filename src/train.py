from streamlit_extras.switch_page_button import switch_page
from experiment import pages_management
import streamlit as st

def main():
    pages_management.update_pages()
    switch_page("Accueil")

def test_function():
    st.write('Hello world Sylvain')
    return 5

if __name__ == "__main__":
    main()
