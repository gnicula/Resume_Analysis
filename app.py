import gradio as gr
import PyPDF2
import os
import gradio as gr
from huggingface_hub import InferenceClient


# client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
# client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3")
client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")


class ResumeJobLearningPlan:

    def __init__(self):
        pass

    def extractTFF(self, file_path):
        # Get the file extension
        file_extension = os.path.splitext(file_path)[1]

        if file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                # Create a PDF file reader object
                reader = PyPDF2.PdfFileReader(file)
                # Create an empty string to hold the extracted text
                extracted_text = ""
                # Loop through each page in the PDF and extract the text
                for page_number in range(reader.getNumPages()):
                    page = reader.getPage(page_number)
                    extracted_text += page.extractText()
                extracted_text = extracted_text.replace('\n', ' ')
                return extracted_text
        elif file_extension == '.txt':
            with open(file_path, 'r') as file:
                # Just read the entire contents of the text file
                return file.read()
        else:
            return "Unsupported file type"

    def format_prompt(self, message, history=None):
        prompt = "<|system|>You are a helpful HR expert, a resume writer expert and a career coach. \
                Be precise, concise, and professional.\n</s>\n"
        prompt += f"<|user|>\n {message} </s>\n<|assistant|>"
        return prompt

    def generate(
        self,
        prompt,
        # history,
        temperature, max_new_tokens=1024, top_p=0.95, repetition_penalty=1.0,
    ):
        temperature = float(temperature)
        if temperature < 1e-2:
            temperature = 1e-2
        top_p = float(top_p)

        generate_kwargs = dict(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            seed=40,
        )

        formatted_prompt = self.format_prompt(prompt)
        print(formatted_prompt)

        stream = client.text_generation(
            formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
        output = ""

        for response in stream:
            output += response.token.text
            yield output

    def modelResponse(self, textjd, textcv, temperature):
        resume = self.extractTFF(textcv)
        job_description = self.extractTFF(textjd)

        response = self.generate(
            f"Given this job description, called $JOBDESC:\n******\n {job_description} \n******\n and this resume called $RESUME:\n******\n {resume} \n******\n, \
                give possible improvements and modifications to the $RESUME to better match the job description. \
                The result should be in this format: \
                Resume Improvements: [Mention the changes to the resume to improve the candidate's chances to get an interview. If there are no matches simply say N/A]. \
                Keywords: [Return all the skills required by the $JOBDESC on a separate line and for each of them write MATCHED if the skill also appears in the $RESUME. If it does not match write NOT MATCHED N/A] \
                Study Guide: [Recommend a list of at most 10 resources (books, websites, courses, Youtube videos) for the candidate to learn skills highly related to the $JOBDESC. \
                Output each resource on a separate line, prioritizing resources for the NOT MATCHED skills in the Keywords section.]",
            temperature,
        )
        return response

    def improvements(self,job_description_path, resume_path, temperature):
        job_description_path = job_description_path.name
        resume_path = resume_path.name
        print(job_description_path, resume_path)

        generated_text = self.modelResponse(job_description_path, resume_path, temperature)

        # result = ''.join([x for x in generated_text]) 
        result = ''
        for x in generated_text:
            result = x
        result = result.replace("</s>", "")
        return result

    def gradio_interface(self):
        with gr.Blocks(css="style.css") as app:
            gr.HTML("""
                <div class="header">
                    <div class="text">
                        <h1>AI Match Resume to Job and Create Learning Plan</h1>
                    </div>
                    <div>
                        <a href="https://www.linkedin.com/in/gabriele-nicula-9847241aa/" target="_blank">
                            <img src="https://media.licdn.com/dms/image/D5635AQG0P-RpyKy1Bg/profile-framedphoto-shrink_400_400/0/1679208113680?e=1721178000&v=beta&t=Xys3eqMruVDmAddXczBMunEfElHcvY56lx0V45ryES8" alt="LinkedIn icon" width="80" height="80">
                        </a>
                    </div>
                </div>
            """)
            with gr.Row():
                with gr.Column(scale=0.5, min_width=160, elem_classes="dark-blue"):
                    jobDescription = gr.File(label="Upload Job Description .txt")
                with gr.Column(scale=0.5, min_width=160, elem_classes="dark-blue"):
                    resume = gr.File(label="Upload Resume .txt or .pdf")
            with gr.Row():
                with gr.Column(scale=0.9, min_width=100, elem_classes="dark-blue"):
                    tempSlider = gr.Slider(label="Model Temperature", minimum=0.0, maximum=1.0, step=0.01, value=0.5)
                with gr.Column(scale=0.10, min_width=150, elem_classes="dark-blue"):
                    analyze = gr.Button("Analyze")
            with gr.Row():
                with gr.Column(scale=1.0, min_width=150, elem_classes="dark-blue"):
                    parts_to_improve = gr.Textbox(label="AI Analysis and Recommendations", lines=25)
            analyze.click(self.improvements, [jobDescription, resume, tempSlider], [parts_to_improve])
                
        app.launch()

resume=ResumeJobLearningPlan()
resume.gradio_interface()
