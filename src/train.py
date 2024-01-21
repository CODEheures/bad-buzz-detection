from streamlit_extras.switch_page_button import switch_page
from experiment import pages_management


def main():
    """Entry point for streamlit Train App
    """
    pages_management.update_pages()
    switch_page("Accueil")


if __name__ == "__main__":
    main()
