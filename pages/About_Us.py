import streamlit as st


def main():
    st.set_page_config(page_title="About LegalGPT", page_icon="❓")

    st.title(":rainbow[About LegalPDF]", anchor = False)

    st.markdown('''
                This is a prototype chatbot that can understand legal documents and answer related queries, developed by students of Year 2 Semester 1 in Bachelor of Computer Science (Honours) in Data Science.
                
                The following are the contributors of this prototype
                - Gan Yee Jing
                - Yeap Jie Shen
                - Jerome Subash A/L Joseph

                This chatbot is powered by **GPT-3.5-Turbo** as the Large Language Model (**LLM**) and **OpenAIEmbeddings** as the text embeddings

                This chatbot latest version is dated at **25<sup>th</sup> December 2023**
                ''', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
