import streamlit as st

#Title

st.markdown("""# Welcome to the GANdy Warhol projet!""")

#How does it work?

st.markdown("""

## How does it work?

GANdy Warhol is a deep learning algorithm which can generate cool abstract expressionism artworks.
Click on the "Generate" button below and let's the magic happen...

""")

#Generate Button

if st.button('GENERATE'):
    # print is visible in the server output, not in the page
    print('button clicked!')
    st.write('I was clicked 🎉')
    st.write('Further clicks are not visible but are executed')
else:
    st.write('I was not clicked 😞')

#GIF Part

st.markdown("""

## Have a look at the algorithm working!

""")

#Comparison with real artwork

st.markdown("""

## Wow, your generated artwork is pretty close to a real one!!

""")

#The project

st.markdown("""

## The project

After 10 superintense weeks at LeWagon, Eloho, YiWen, Saul and Flavien designed this Generative Adversarial Algorithm in 2 weeks.
It has been trained on more than 3,500 images and is now able to create new art pieces.

""")

#Photo of the team

st.markdown("""

## It's us!

""")


CSS = """
h1 {
    color: black;
}
.stApp {
    background-image: url(https://i.insider.com/5501ab9c6da811ed108dbfb3?width=1136&format=jpeg);
    background-size: cover;
}
"""

if st.checkbox('Inject CSS'):
    st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)
