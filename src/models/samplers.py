class ProFamSampler:
    def __init__(self, name, model, prompt_builder, sampling_kwargs):
        self.name = name
        self.model = model
        self.prompt_builder = prompt_builder
        self.model_kwargs = sampling_kwargs

    def sample_seqs(self, protein_document, num_samples):
        prompt = self.prompt_builder.build_prompt(protein_document)
        inputs = self.prompt_builder.build_inputs_from_prompt(prompt, num_samples)
        return self.model.sample_seqs(**inputs, **self.sampling_kwargs)
