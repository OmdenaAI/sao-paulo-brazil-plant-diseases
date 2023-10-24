import streamlit as st
import pandas as pd
from PIL import Image
from helping_functions import load_css
# st.set_page_config(page_title='The Team', layout='wide')
st.set_page_config(page_title='The Team', page_icon=':herb:', initial_sidebar_state='auto',  layout='wide')

load_css()

fa_li = """
<i class="fab fa-linkedin"></i>
"""
fa_em = """
<i class="far fa-envelope"></i>
"""

col1, col2 = st.columns((1, 5))
st.subheader('Our Team')
st.write('Omdena Sao Paulo chapter was made up of a team from over 10 different timezones. \
    Despite the distances and time separations we manged to get together and successfully \
    collaborate and produce some excellent work.')
st.write('All involved played an important role to the outcomes of the project.')

col1, col2, col3, col4 = st.columns((1, 1, 1, 1))

IMAGE_SIZE = (128, 128)

with col1:
    wallace = Image.open('assets/Images/foto_linkedin_2 - Wallace Ferreira.jpg').resize(IMAGE_SIZE)
    st.image(wallace)
    st.header('Wallace G. Ferreira')
    st.subheader('Task 1: Research Knowledge Brainstorming')
    st.subheader('Chapter Lead')
    st.write('"I am a data scientist with a passion for continuous learning. I am deeply interested in leveraging the power of science, technology, and artificial intelligence to make a positive impact on society. I am Chapter Lead at Omdena São Paulo Chapter."')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/wallace-ferreira/](https://www.linkedin.com/in/wallace-ferreira-69065910/)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/wallacegferreira](https://github.com/wallacegferreira)',
        unsafe_allow_html=True
    )

with col2:
    hari = Image.open('assets/Images/hari - Hariharan Ayappane.jpg').resize(IMAGE_SIZE)
    st.image(hari)
    st.header('Hariharan')
    st.subheader('Task 8: Webapp Deployment')
    st.subheader('Task Lead')
    st.write('Hey there! I have a diverse set of experiences ranging from embedded software/hardware, web development and ML/DL/LLM development. I strongly believe all these experiences can come together to produce a meaningful contribution to the world and improve the lives of people via technology and community.')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/hariharan-ayappane](https://www.linkedin.com/in/hariharan-ayappane)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/haran2001](https://github.com/haran2001)',
        unsafe_allow_html=True
    )


with col3:
    TB = Image.open('assets/Images/FB_IMG_1542851662878 - Tácio Barros.jpg').resize(IMAGE_SIZE)
    st.image(TB)
    st.header('TÃ¡cio Fonseca Barros')
    st.subheader('Task 9: Final Paper')
    st.subheader('Task co-lead')
    st.write('"Task 2: worked on models to increas  image size and quality \n Task 9: literature review and article writing and organization"')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/tacio-barros](https://www.linkedin.com/in/tacio-barros)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/TassHub](https://github.com/TassHub)',
        unsafe_allow_html=True
    )
    
with col4:
    lucas = Image.open('assets/Images/eu_cortado - Lucas Vasconcelos.png').resize(IMAGE_SIZE)
    st.image(lucas)
    st.header('Lucas Vasconcelos Rocha')
    st.subheader('Task 4: Models Building')
    st.subheader('Team Member')
    st.write('"I am a Data Scientist with significant experience (+3 yrs) working with machine learning and data science projects.\n I was working on tasks for preparing and processing the data, organizing brainstorms and meetings to discuss the project\'s quality, and working on the model building."')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/lucasvasconcelosrocha](https://www.linkedin.com/in/lucasvasconcelosrocha/)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/lucasvascrocha](https://github.com/lucasvascrocha)',
        unsafe_allow_html=True
    )


with col1:
    juan = Image.open('assets/Images/JCO2 - Juan Olano.jpeg').resize(IMAGE_SIZE)
    st.image(juan)
    st.header('Juan Olano')
    st.subheader('Task 4: Models Building')
    st.subheader('Team Member')
    st.write('I developed a CNN and trained from scratch. I trained it with each one of the 2 strategies, 1 and 2.  At the time the results were very poor, as happened to other models built and trained by other participants. \n I also developed a StyleGAN to synthesize images.')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[www.linkedin.com/in/juan-olano](https://www.linkedin.com/in/juan-olano-b9a330112/)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/jcolano](https://github.com/jcolano)',
        unsafe_allow_html=True
    )


with col2:
    ss = Image.open('assets/Images/WhatsApp Image 2023-07-25 at 12.50.18 PM - Sourabh Singhal.jpeg').resize(IMAGE_SIZE)
    st.image(ss)
    st.header('SOURABH SINGHAL')
    st.subheader('Task 2: Data Collection Preprocessing')
    st.subheader('Team Member')
    st.write('Given research papers and datasets of mongo leaf and different plants. ')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/sourabh-singhal](https://www.linkedin.com/in/sourabh-singhal-092549203)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/saasfasterboy](https://github.com/saasfasterboy)',
        unsafe_allow_html=True
    )

with col3:
    tamiris = Image.open('assets/Images/foto2_t - Tamiris Crepalde.jpg').resize(IMAGE_SIZE)
    st.image(tamiris)
    st.header('Tamiris Crepalde')
    st.subheader('Task 2: Data Collection Preprocessing')
    st.subheader('Team Member')
    st.write('I worked mainly in the Data Collection and Pre-processing phase. I helped collect data from some sources and developed deduplication code, which helped create a set of datasets without repeated images. Furthermore, I have participated in discussions about strategies and best practices for model development, aiming to avoid data leakage and guarantee the quality and reliability of the model.')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/tamiriscrepalde/](https://www.linkedin.com/in/tamiriscrepalde/)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/tamiriscrepalde](https://github.com/tamiriscrepalde)',
        unsafe_allow_html=True
    )

with col4:
    leul = Image.open('assets/Images/omd - leul assaminew.jpg').resize(IMAGE_SIZE)
    st.image(leul)
    st.header('Leul Assaminew')
    st.subheader('Task 4: Models Building')
    st.subheader('Team Member')
    st.write('Model building and fine tuning model, model deployment')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/leul-assaminew](https://www.linkedin.com/in/leul-assaminew)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/leulassaminew](https://github.com/leulassaminew)',
        unsafe_allow_html=True
    )

with col1:
    swati = Image.open('assets/Images/SRZPH - Swati Zambre.jpg').resize(IMAGE_SIZE)
    st.image(swati)
    st.header('Swati Zambre')
    st.subheader('Task 6: Models Fine Tuning Validation')
    st.subheader('Task Lead')
    st.write('I was committed to research brainstorming and paper writing process throughout the process.')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/swatizambre/](https://www.linkedin.com/in/swatizambre/)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[swatizambre](swatizambre)',
        unsafe_allow_html=True
    )


with col2:
    anastasiia = Image.open('assets/Images/avatar - Anastasya Marchenko.jpg').resize(IMAGE_SIZE)
    st.image(anastasiia)
    st.header('Anastasiia Marchenko')
    st.subheader('Task 8: Webapp Deployment')
    st.subheader('Team Member')
    st.write('Develop and deploy a Streamlit application. ')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/anastasia-marchenko](https://www.linkedin.com/in/anastasia-marchenko)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/dots13](https://github.com/dots13)',
        unsafe_allow_html=True
    )


with col3:
    aiman = Image.open('assets/Images/passport_size_photo - Aiman Lameesa.jpeg').resize(IMAGE_SIZE)
    st.image(aiman)
    st.header('Aiman Lameesa')
    st.subheader('Task 9: Final Paper')
    st.subheader('Team Member')
    st.write('Contributed to the literature review part')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/aiman-lameesa](https://www.linkedin.com/in/aiman-lameesa/)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/aimanlameesa](https://github.com/aimanlameesa)',
        unsafe_allow_html=True
    )

with col4:
    dimitra = Image.open('assets/Images/digital-photo - Dimitra Muni.JPG').resize(IMAGE_SIZE)
    st.image(dimitra)
    st.header('Dimitra Muni')
    st.subheader('Task 7: Webapp Development')
    st.subheader('Task Lead')
    st.write('#NAME?')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/dimitramuni](https://www.linkedin.com/in/dimitramuni)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/dimitramuni](https://github.com/dimitramuni)',
        unsafe_allow_html=True
    )


with col1:
    paula = Image.open('assets/Images/foto - Paula Carolina Montagnana.jpg').resize(IMAGE_SIZE)
    st.image(paula)
    st.header('Paula Montagnana')
    st.subheader('Task 9: Final Paper')
    st.subheader('Task Lead')
    st.write('I coordinated the writing task and also participated in writing the paper reporting the activities developed during the project, for publication in the journal \'Omdena Journal of Artificial Intelligence for Impact\'. I adjusted a CNN model using the R programming language, \'tensorflow\' package.')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/paula-montag](https://www.linkedin.com/in/paula-montag/)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/p-montagnana](https://github.com/p-montagnana)',
        unsafe_allow_html=True
    )

with col2:
    ksenia = Image.open('assets/Images/IMG_5151 - Kseniya Trusova.JPG').resize(IMAGE_SIZE)
    st.image(ksenia)
    st.header('Ksenia Trusova')
    st.subheader('Task 9: Final Paper')
    st.subheader('Task co-lead')
    st.write('Found two articles for the Research task. One of them described two most popular public datasets PlantVillage and PlantDoc. Processed the articles found by the team for the Final Paper and created a literature overview with important details about the findings of other researchers in order to compare the results achieved by the team.')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/kseniya-trusova-bitdn/](https://www.linkedin.com/in/kseniya-trusova-bitdn/)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/kseniya-trusova](https://github.com/kseniya-trusova)',
        unsafe_allow_html=True
    )

with col3:
    whaner = Image.open('assets/Images/whaner_mam - Whaner Endo.jpg').resize(IMAGE_SIZE)
    st.image(whaner)
    st.header('Whaner Endo')
    st.subheader('Task 2: Data Collection Preprocessing')
    st.subheader('Team Member')
    st.write('I collaborated with the project kick off, brainstorming about the problem statment and then with the next phase, helping the data collection, working with the EMBRAPA dataset. Attended several meetings and discussed the final paper.')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/whaner](www.linkedin.com/in/whaner)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/whaner](www.github.com/whaner)',
        unsafe_allow_html=True
    )
    


with col4:
    st.header('Maria Fisher')
    st.subheader('Task 1: Research Knowledge Brainstorming')
    st.subheader('Task Lead')
    st.write('Assisting with background knowledge in plant pathogen, computer vision and writing a scientific article')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/marÉ¨a-fÉ¨sher](https://www.linkedin.com/in/marÉ¨a-fÉ¨sher)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/maria-fisher](https://github.com/maria-fisher)',
        unsafe_allow_html=True
    )



with col1:
    st.header('Darshan Pai')
    st.subheader('Task 1: Research Knowledge Brainstorming')
    st.subheader('Task co-lead')
    st.write('Decide on the problem formulation in the first week. Then I helped curate and augment the data for modeling. Tried a couple of modeling techniques and shared the results with the team.')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/darshanpai](https://www.linkedin.com/in/darshanpai)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/dpai](www.github.com/dpai)',
        unsafe_allow_html=True
    )


with col2:
    st.header('Kilian BÃ¤nziger')
    st.subheader('Task 1: Research Knowledge Brainstorming')
    st.subheader('Task Lead')
    st.write('I coordinated a feasibility study to use drone images for the detection of sick plantation areas in coffee plantations ')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/kilian-b%C3%A4nziger-9488581b2/](https://www.linkedin.com/in/kilian-b%C3%A4nziger-9488581b2/)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/kilibenz](https://github.com/kilibenz)',
        unsafe_allow_html=True
    )


with col3:
    st.header('Jebathurai I Barnabas')
    st.subheader('Task 1: Research Knowledge Brainstorming')
    st.subheader('Team Member')
    st.write('After thorough planning and brainstorming, I have meticulously outlined a data science project, considering various aspects such as data sources, methodologies, and potential outcomes.')
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/jebathurai-barnabas](https://in.linkedin.com/in/jebathurai-barnabas)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[https://github.com/Jebathurai-JB][https://github.com/Jebathurai-JB)',
        unsafe_allow_html=True
    )

with col4:
    sabelo = Image.open('assets/Images/Mbatha 2 - Sabelo Mbatha.jpg').resize(IMAGE_SIZE)
    st.image(sabelo)
    st.header('Sabelo Mbatha')
    st.subheader('Task 5: Models Improvement')
    st.subheader('Task co-lead')
    st.write("- Created folders for the team, and uploaded problem statements to the Google Shared drive. Introduced and started work on the trello board and how to anwer the big picture questions of our data. Checked and curated a lot of datasets on the Trello board so to get the Coffee dataasets we need and put aside we don't need as yet. Collected 11 UAV image datasets/data sources for the Remote sensing task, curated and checked to make sure we have data about coffee. Came up with and lead a team exercise that will enhance and promote Trust and belief in each other (the Huddle) and had a meet with a few collaborators. Merged some Coffee datasets we needed. Started and lead a first draft of the \"introduction\" for the Final writeup. Followed up on collaborators to submit work for final write up. Created a Template to report work that has been done by collaborators and each Task. Created and drafted a Template to Report on all the models that have been done")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/sabelombatha-baptiste](linkedin.com/in/sabelombatha-baptiste)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/sabunity][https://github.com/sabunity)',
        unsafe_allow_html=True
    )

with col3:
    st.header('Alpha Lossangoyi-Nanga')
    st.subheader('Task 3: Data Augmentation')
    st.subheader('Task Lead')
    st.write("As Task Lead, I organised and steered the Data Augmentation Task. I collected the multiple proposals of the contributors. I made all the contributors agree on a written and concrete battle plan for the Data Augmentation task. This battle plan consisted of a breakdown of all the necessary subtasks of the Data Augmentation task. The subtasks were discussed, documented, prioritised and assigned. During the weekly meetings, the contributors reported the progress of their assigned subtasks. We updated our battle plan with task statues.I coordinated with the previous \(Data Collection\) and subsequent \(Data Modelling\) Tasks.")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write(
        fa_li,
        '[linkedin.com/in/alphalossangoyi](linkedin.com/in/alphalossangoyi)',
        unsafe_allow_html=True
    )
    st.write(
        fa_li,
        '[github.com/AlphaLossangoyiNanga2][https://github.com/AlphaLossangoyiNanga2)',
        unsafe_allow_html=True
    )
