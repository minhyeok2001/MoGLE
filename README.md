# MoGLE: Mixture of Genre LoRA Experts

MoGLE is a MoE–based architecture designed to enhance genre-specific immersion of LLMs for Tabletop Role-Playing Games.

While LLM-based GMs enable personalized and always-available storytelling, 

they often suffer from genre inconsistency, where narrative tone unexpectedly drifts (see the details in ai-intensive-final.pdf)

By dynamically mixing multiple genre-specialized LoRA adapters, 

MoGLE enables both **stable genre adherence** and also the **smooth mid-story genre transitions**.

--- 
## Architecture

We inject 6 LoRA per genre model to Llama 3.1 8B Instruct and mix them with gating function.

<p align="center">
   <img width="500" height="400" alt="image (2)" src="https://github.com/user-attachments/assets/80deaf6c-e312-4aff-8867-8960cf710ea6" /><br>
   <i>Model Architecture</i>
</p>

There are Two-Phase Training Pipeline to train this model, (1) Single-LoRA Specialization, (2) Gating Function Training.

--- 

### 1. Single-LoRA Specialization

Each genre-specific LoRA is trained independently to specialize in its own narrative style.

We have 5 different genres in our dataset, So 5 different LoRAs are expected to be trained.

<p align="center">
   <img width="400" height="400" alt="image (3)" src="https://github.com/user-attachments/assets/a16484fc-826d-41c9-a17d-5b49e1553508" /><br>
   <i>Single LoRA Specialization Phase</i>
</p>

---

### 2. Gating Function Training

In this phase, we freeze both the base model and all LoRA experts, and train only the gating function. 

The gate learns to dynamically weight and combine genre experts according to the input context, 

allowing MoGLE to maintain distinct genre styles while supporting flexible expert mixing during inference without explicitly specifying genre.

<p align="center">
   <img width="500" height="400" alt="image (4)" src="https://github.com/user-attachments/assets/5fcc5df9-dd94-46c1-96c9-5fb81bf0d077" /><br>
   <i>Single LoRA Specialization Phase</i>
</p>


## Evaluation Pipeline

We evaluate MoGLE using three complementary methods

**1. SOTA Similarity Comparison**

<p align="center">
   <img width="600" height="450" alt="image (5)" src="https://github.com/user-attachments/assets/870ba009-ac4d-479e-96bb-53d9944fd282" /><br>
   <i>SOTA Similarity Comparison Pipeline</i>
</p>

Compared against models such as Llama-4 Maverick (17B) and GPT-OSS (120B) as SOTA model !
  
We use e5-large embeddings as a widely adopted SOTA baseline to measure semantic similarity with large reference models. 

Although e5-large is not assumed to be an optimal metric for narrative evaluation, 

it serves as a reasonable minimum standard for semantic alignment. 

To specifically assess genre and writing style consistency, we further apply Style-Distance, 

which focuses on stylistic similarity beyond content-level semantics.

--- 

**2. LLM Judge Evaluation**

For LLM-based evaluation, we assess genre appropriateness and genre transition quality using an LLM judge. 

The evaluation criteria are derived from Wikipedia-defined genre characteristics, 

providing a standardized and human-interpretable reference for genre definitions. 

Detailed LLM judging prompts are included in the appendix of ai-intensive-final.pdf.

--- 

**3. Genre Classifier**
   
<p align="center">
   <img width="188" height="284" alt="image (6)" src="https://github.com/user-attachments/assets/bfc69608-f9b1-484b-8e91-593eeaa86bcb" /><br>
   <i>Genre Classifier</i>
</p>

As an additional quantitative signal, we use a fine-tuned LongFormer-based genre classifier 

that outputs the probability of the generated text belonging to the target genre

--- 

## Result

<!-- Eval 1: Genre-Appropriate Generation (Single Genre) -->
<h3>Eval 1: Genre-Appropriate Generation</h3>
<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>SOTA comparison - e5</th>
      <th>SOTA comparison - sd</th>
      <th>LLM judge</th>
      <th>Genre classifier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Base (finetune x)</td>
      <td>0.930</td>
      <td><b>0.99145</b></td>
      <td>0.5375</td>
      <td>0.5294</td>
    </tr>
    <tr>
      <td>Base x 5 (finetune per genre)</td>
      <td><b>0.93196</b></td>
      <td>0.98408</td>
      <td><b>0.64418</b></td>
      <td><b>0.69874</b></td>
    </tr>
    <tr>
      <td>Base (Huge lora)</td>
      <td>0.92955</td>
      <td><u>0.98415</u></td>
      <td>0.32165</td>
      <td>0.6221</td>
    </tr>
    <tr>
      <td>MoGLE (ours, 0.0)</td>
      <td><u>0.93086</u></td>
      <td>0.9836</td>
      <td><u>0.62084</u></td>
      <td><u>0.652985</u></td>
    </tr>
  </tbody>
</table>

<br/>

<!-- Eval 2: Genre-Shift Adaptation (Dynamic Transition) -->
<h3>Eval 2: Genre-Shift Adaptation</h3>
<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>SOTA comparison - e5</th>
      <th>SOTA comparison - sd</th>
      <th>LLM judge</th>
      <th>Genre classifier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Base (finetune x)</td>
      <td>0.864820</td>
      <td>0.848823</td>
      <td>0.552842</td>
      <td>0.642815</td>
    </tr>
    <tr>
      <td>Base (Huge lora)</td>
      <td><u>0.873102</u></td>
      <td><u>0.850192</u></td>
      <td><u>0.561242</u></td>
      <td><u>0.651128</u></td>
    </tr>
    <tr>
      <td>MoGLE (ours, 0.1)</td>
      <td><b>0.880365</b></td>
      <td><b>0.860535</b></td>
      <td><b>0.588335</b></td>
      <td><b>0.908030</b></td>
    </tr>
  </tbody>
</table>

---

## Conclusion

MoGLE maintains strong genre consistency while fluidly adapting to mid-story genre transitions, enabling more immersive and controllable TRPG narratives.

In genre-appropriate generation, MoGLE does not suffer from performance degradation despite using significantly fewer parameters compared to per-genre fine-tuning approaches (e.g., Base ×5). 

This indicates that dynamic expert mixing can preserve genre fidelity without relying on costly, fully specialized models.

In genre-shift adaptation, MoGLE consistently outperforms strong baselines, 

demonstrating its effectiveness in handling dynamic genre transitions—one of the core challenges in LLM-based TRPG systems.

Nevertheless, our evaluation is conducted on a relatively limited and narrowly scoped dataset, which constrains the confidence and generality of the observed performance gains. 

As a result, while the trends are consistent, the absolute improvements should be interpreted with caution. 

We expect that scaling to larger and more diverse datasets will further clarify and potentially amplify the advantages of MoGLE, particularly in complex genre-transition scenarios.


**You can find the full project details in ai-intensive-final.pdf.**
