import openai

openai.api_type = "azure"
openai.api_base = "https://emailgeneratordemo.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "4461d4ebc79a45bca18557145962a4f3"

 # Send request to Azure OpenAI model
print("Sending request for summary to Azure OpenAI endpoint...\n\n")
models=openai.Model.list()

chat_completion = openai.ChatCompletion.create(deployment_id="EmailGeneratorDemo01",
    model="gpt-35-turbo", messages=[{"role": "user", "content": "Hello world"}])
print(chat_completion.choices[0].message.content)


response = openai.ChatCompletion.create(
     deployment_id="EmailGeneratorDemo01",
     temperature=0.7,
     max_tokens=120,
     messages=[
         {"role": "system", "content": "You are a helpful assistant. Summarize the following text in 60 words or less."}
     ]
 )

print("Summary: " + response.choices[0].message.content + "\n")