from addict import Dict
ori_lst = [
    ["S181", "PAR: THAT'S A GOOD WAY TO BREAK HIS NECK"],
    ["S181", "PAR: BREAK HIS BACK I SHOULDA SAID"],
    ["S205", "PAR: THE LITTLE GIRL IS LAUGHING AT THE BOY FALLING OFF THE CHAIR"]
]

rep_lst = [
    ["S181", "PAR: THAT WILL HURT HIMSELF"],
    ["S181", "PAR: LIKE HURT HIS BACK I SHOULDA SAID"],
    ["S205", "PAR: THE LITTLE GIRL FINDS AMUSEMENT IN THE BOY ACCIDENTALLY FALLING OFF THE CHAIR"]
]

Sensitive_replace_dict = {}

for ori_item, rep_item in zip(ori_lst, rep_lst):
    key = ori_item[0]
    value = (ori_item[1], rep_item[1])
    if key not in Sensitive_replace_dict:
        Sensitive_replace_dict[key] = []
    Sensitive_replace_dict[key].append(value)
#======================================================    
mmse_people_select={}
mmse_people_select['mmse_low']=set(['S111','S090'])
mmse_people_select['mmse_middle']=set(['S110','S114'])
mmse_people_select['mmse_high']=set(['S061','S062'])

mmse_analyze_selected_people = set()

# 合併所有值到 selected_people 集合中
for value_set in mmse_people_select.values():
    mmse_analyze_selected_people = mmse_analyze_selected_people.union(value_set)
#======================================================
Symbol_template={}
Symbol_template["importanceTag"]={
    'st':'[notice]',
    'ed':'[\\notice]'
}
Psychology_template=Dict()

# Psychology_template['anomia']['definition']=f"Empty speech,  trailing off speech, circumlocution in speech"
# Psychology_template['anomia']['example']=[f"Empty speech: He’s trying to get {Symbol_template['importanceTag']['st']} this {Symbol_template['importanceTag']['ed']} and he’s gonna fall off of {Symbol_template['importanceTag']['st']} there {Symbol_template['importanceTag']['ed']}",
#                                           f"trailing off speech: If that little girl {Symbol_template['importanceTag']['st']} don’t xxx {Symbol_template['importanceTag']['ed']}",
#                                           f"circumlocution in speech: The boy hasn’t {Symbol_template['importanceTag']['st']} gotten down to his {Symbol_template['importanceTag']['st']} fall {Symbol_template['importanceTag']['ed']} yet."]

Psychology_template['anomia']['definition']=f"Empty speech,  trailing off speech, circumlocution in speech"
Psychology_template['anomia']['example']=[f"Empty speech: Eloquent articulation lacking the expression of meaningful information.",
                                          f"trailing off speech: dropping speech, when the last few words in an utterance become barely audible.",
                                          f"circumlocution in speech: circumlocution of words/concepts within an utterance."]


Psychology_template['disflueny']['definition']=f"Word/phrase revision, word/phrase repetition, phonological fragment"
Psychology_template['disflueny']['example']=[f"Word/phrase revision: The wife is wiping a {Symbol_template['importanceTag']['st']} dish plate. {Symbol_template['importanceTag']['ed']}",
                                             f"word/phrase repetition: {Symbol_template['importanceTag']['st']} His his {Symbol_template['importanceTag']['ed']} sister’s asking for one.",
                                             f"phonological fragment: Here’s a {Symbol_template['importanceTag']['st']} sp {Symbol_template['importanceTag']['ed']} water spigot here."
                                             ]

Psychology_template['Agrammatism']['definition']=f"Telegraphic speech, misuse of pronuns, poor grammar"
Psychology_template['Agrammatism']['example']=[f"Telegraphic speech: {Symbol_template['importanceTag']['st']} Water running down {Symbol_template['importanceTag']['ed']} from the sink.",
                                               f"misuse of pronuns: {Symbol_template['importanceTag']['st']} Her doing {Symbol_template['importanceTag']['ed']} the dishes.",
                                               f"poor grammar: Three pieces {Symbol_template['importanceTag']['st']} of to eat on {Symbol_template['importanceTag']['ed']}."]


# Content and Semantic Deficits
Psychology_template['scene_difficulty'] = {
    'definition': "Difficulty describing the scene, omitting important elements or providing inaccurate details, reflecting semantic memory deficits.",
    'example': [
        "Empty Speech: He’s trying to get [his coat] this [morning], and he’s gonna fall off of [the bed] there.",
        "Trailing Off Speech: If that little girl [doesn’t pick up] the [toys], then she’ll be [in trouble].",
        "Circumlocution in Speech: The boy hasn’t [finished] his [homework] yet."
    ]
}

# Word Finding and Vocabulary Impairments
Psychology_template['hesitation_and_pauses'] = {
    'definition': "Hesitation and pauses in speech, experiencing difficulty finding appropriate words, leading to pauses or circumlocutions.",
    'example': [
    "Hesitation and Pauses: The individual tried to describe a common household tool used for tightening screws but hesitated, saying, 'I need that thing, you know, the one with a handle that turns and helps put things together.'"
    ]

}

Psychology_template['limited_vocabulary'] = {
    'definition': "Limited vocabulary, using simpler or more generic terms instead of specific words, indicating a decline in vocabulary.",
    'example': [
        "Limited Vocabulary: The elderly man struggled to express his desire for a hot beverage and simply stated, 'I want a, um, warm drink like tea, but not tea, you know, the one people have in the morning.'",
        ]
}

# Syntactic and Grammatical Impairments
Psychology_template['sentence_construction_issues'] = {
    'definition': "Sentence construction issues, trouble constructing grammatically correct sentences, leading to simpler and less complex sentence structures.",
    'example': [
        "Sentence Construction Issues: He, um, went to the store and, uh, bought some, you know, groceries for dinner.",
        "Simplified Sentence Structure: The movie was, uh, good. It had, you know, action and, um, interesting characters."
    ]
}


Psychology_template['impaired_syntax'] = {
    'definition': "Impaired syntax, syntax errors such as incorrect word order or tense in narrative.",
    'example': [
        "Impaired Syntax: Yesterday, I was, you know, meeting my friend for lunch, and we, uh, talked about the, um, upcoming event.",
        ]
}

# Pragmatic Language Deficits
Psychology_template['lack_of_narrative_coherence'] = {
    'definition': "Lack of narrative coherence, struggling to organize description in a logical and coherent manner, disrupting the flow of the narrative.",
    'example': ["Lack of Narrative Coherence: So, um, there was this, you know, place, and, uh, people were doing things, but I can't quite, you know, remember how it all fits together.",
 "Simplified Sentence Structure: The movie was, uh, good. It had, you know, action and, um, interesting characters.",
 "Difficulty Organizing Description: I saw, um, a thing, and it was, you know, interesting because of, uh, some reasons that I can't quite, you know, put in order."
 ]
}

# Comprehension Deficits
Psychology_template['misinterpretation_of_details'] = {
    'definition': "Misinterpretation of details, misunderstanding or misinterpreting elements in a picture, reflecting comprehension difficulties.",
    'example': [
        "Misinterpretation of Details: The, you know, person in the picture was, um, doing something strange, like, um, dancing, but I'm not sure exactly what.",
        ]
}

Psychology_template['inability_to_answer_questions'] = {
    'definition': "Inability to answer questions about a scene, struggling to provide accurate and relevant answers, indicating impaired comprehension.",
    'example': [
        "Inability to Answer Questions: I, um, don't really know what's happening in the picture. You see, there are, you know, things, but I can't say much about them.",
        "Difficulty Providing Relevant Answers: When you asked about, um, the scene, I'm not sure, you know, what to say. It's a bit confusing for me." 
        ]
}

# Memory Impairments
Psychology_template['limited_recall_of_details'] = {
    'definition': "Limited recall of details, difficulty recalling specific details from a picture, reflecting memory deficits.",
    'example': [
        "Limited Recall of Details: I remember, you know, seeing something, but I can't recall the, uh, specific details, like colors or, um, what people were doing.",
        ]
}

Psychology_template['repetition_errors'] = {
    'definition': "Repetition errors, struggling to recall information accurately when a picture is shown again, demonstrating short-term memory challenges.",
    'example': [
        "Repetition Errors: Oh, this picture looks familiar, but, um, I can't quite remember what I said the last time. Sorry, my, you know, memory isn't working well.",
        ]
}

# Prosodic and Articulatory Features
Psychology_template['altered_prosody'] = {
    'definition': "Altered prosody, changes in intonation, rhythm, or stress patterns of speech, impacting the natural flow of the narrative.",
    'example': [
        "Altered Prosody: So, I was, um, at the park, and there was this, you know, bird singing, and it was like, uh, tweeting, but not in a regular way.",
        ]
}

Psychology_template['dysarthria'] = {
    'definition': "Dysarthria, difficulties in articulating sounds clearly, leading to changes in speech clarity.",
    'example': [
        "Dysarthria: I, um, wanted to tell you about my, you know, day, but my words are, uh, not coming out very clearly. It's like, um, they're getting stuck.",
 "Changes in Speech Clarity: The, uh, way I'm talking might sound a bit, you know, different. It's like, um, my mouth isn't forming the words quite right."
 
        ]
}



Instruction_templates={}
Instruction_templates['psychology']=["""Read the psycological definition above fisrt. Please analyze the Dialogue provided below, and check if PAR has problems appeared in any of the psycological definitions. Finally, provide a summary about the PAR's speech.
* If you didn't detect any problem please leave it blank. 
* You should only analyze what appeared in the dialogue, and do not analyze anything appear in the psycological definition
* Keep the summary short and precise. 
* The answer should be in the format of:

detected problems:
Summary: 

Before starting, lets see one answer example about other dialogue (not dialogue in this case):
---
detected problems:
- Empty speech: PAR's utterances like "OKAY" and "I THINK THAT'S IT" do not convey meaningful information.
- Trailing off speech: PAR's utterance "OKAY" trails off as it is cut off abruptly.
- Circumlocution in speech: PAR's utterance "THE GIRL HAS LOOKS LIKE SHE'S LAUGHING" shows circumlocution by using the phrase "LOOKS LIKE" instead of directly stating that the girl is laughing.

Summary: PAR's speech includes instances of empty speech, trailing off speech, and circumlocution, indicating a lack of expression of meaningful information and a tendency to avoid direct and concise communication."""]


assesmentPrompt_template=Dict()
assesmentPrompt_template['basic']="""psycological definition:
{psychology_template}


{Instruction_template}

Dialogue:

{content}


"""

##********************** TBD: external_source 怎麼找要找一下
assesmentPrompt_template['external_source']="""{Instruction_template}

psycological definition:
---

{psychology_template}

{external_source}

Dialogue:

- {content}


"""

assesmentPrompt_template['data_augmentation']="""Based on the following dementia assessment results: 

{content}

can you help keep the speaker's style and Alzheimer's disease status and generate another description results to simulate the patient performing the test again?"""


def generate_psychology_prompt(assessment_prompt_template, instruction_templates, psychology_template):
    prompts_dict = {}

    for instruction_template in instruction_templates['psychology']:
        for key, values in psychology_template.items():
            definition = psychology_template[key]['definition']
            examples = '\n'.join(psychology_template[key]['example'])

            psychology_temp = f"""
            - definition: {definition}
            - examples: {examples}
            """

            assess_prompt_kargs = {
                'Instruction_template': instruction_template,
                'psychology_template': psychology_temp,
                'content': '{dialogue_content}'
            }
            prompt = assessment_prompt_template['basic'].format(**assess_prompt_kargs)
            prompts_dict[key]=prompt

    return prompts_dict
# Usage:
# result_prompts = generate_psychology_prompt(assessment_prompt_template=assesmentPrompt_template,
#                                             instruction_templates=Instruction_templates,
#                                             psychology_template=Psychology_template,
#                                             )
