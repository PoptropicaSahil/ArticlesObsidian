

## **Building Natural Language Search at Zomato**

- “Veg Chinese without garlic” 
- “Fever food under 400” 
- “Something light for an upset stomach” 
- “soothing food for a sore throat” → returns whatever happens to contain the word “throat”

> Keyword search isn’t sufficient

Natural Language Search (NLS):
- INPUT - real, messy human sentence
- OUTPUT - dishes and restaurants 

## **Two very different ways people search**

Eeal queries split cleanly into two intents that pull in opposite directions:

- **Known-item search :** the customer already knows what they want and is just navigating to it: “Starbucks”, “Domino’s”, “piz”. Here **speed and precision matter most**; the “right” answer is a specific, high-confidence entity.
- **Intent / exploratory search :** the customer describes a **need** rather than a name: “Comfort food” ,“Light meals”, “Netflix and chill”, “my favourites”. Here **understanding** matters most; there is no single right answer, only a set of relevant ones.

Optimising for one tends to hurt the other. A system tuned purely for semantic understanding feels sluggish and over-thinks a plain “Domino’s”; a system tuned purely for keyword speed is utterly lost on “fever food.”
> NLS runs **both** strategies in parallel and merges them at the end

## The architecture at a glance

NLS follows a **parallel, hybrid-search** design. When a query arrives, the **search backend** hands it to a lightweight **Splitter** that fans the request out into two independent streams which run at the same time:

- **Stream A : Entity Retrieval (Traditional Search):** fast, keyword-based entity retrieval via Apache Solr, ranked by a tuned Learning-to-Rank (LTR) model
- **Stream B : Intent & Semantic Search (GenAI + Vector Search):**  LLM that decodes what the customer actually means, followed by a fine-tuned embedding model that maps that meaning onto real dishes on the platform.

The two streams converge at a **Result Aggregation** layer, where everything is re-ranked for hyper-personalisation before it reaches the customer

![](Safari%20-%2027%20Oct%202026%20at%205_40%20AM_page-0001.jpg)

### Stream A - the fast lane: Solr + Learning to Rank

When the customer is almost certainly reaching for a specific, known entity - a dish, a cuisine, or a restaurant.

- **Solr (entity retrieval) :** get candidate entities using lexical similarity - the classic, extremely fast *inverted-index match*
- **LTR :** scores candidates to balance pure relevance against business metrics 
- **Top@K :** only top 3–4 high-confidence entities → if someone types “Starbucks”, it should appear instantly

### Stream B - the thinking lane: intent, then meaning

**LLM** interprets intent + **bi-encoder** grounds that intent in real catalogue

### The decoding engine (LLM)

Self-hosted Gemma. Operates in **two modes**, switching automatically based on how specific the query is.

#### **NER Mode - for structured queries**
When a *query names a dish or restaurant along with filters* → LLM as entity extractor. For _“spicy veg noodles under 500, no garlic”_ 

```json
{
  "dish": ["noodles"],
  "taste_profile": "spicy",
  "price": "<500",
  "dietary_preferences": "veg",
  "exclusion_ingredient": "garlic"
}
```

#### **Generative Mode - for vague, needs-based queries.**
When the *query describes a situation rather than a dish* → model generates candidate dishes with rationale . For _“fever food under 400”_

```json 
{
  "dish": ["khichdi", "daliya", "soup"],
  "price": "<400",
  "exclusion_ingredient": "spices, oily",
  "reason": "suggesting easy-to-digest dishes to help you recover faster."
}
```

**The retrieval engine (Vector Search)**
LLM results are ideas (“khichdi”, “clear soup”) not rows in our database.  Retrieval engine maps those ideas → the actual catalogue dishes 

- **Model :** a fully fine-tuned sentence-transformers/paraphrase-mpnet-base-v2 (MPNET), trained on Zomato’s own data
> _paneer tikka_ NOT close to _paneer tikka masala_: lexically close, but a dry starter is not a gravy main

- **Index :**  in-memory HNSW graph with ANN retrieval

### **Bringing the two streams together**

Re-rank the combined pool for **hyper-personalisation** - including individual customer's context and history

### **How we tuned the intent brain (the LLM)**

**The task.**

> Given a natural-language query, extract or generate the entities (dishes, restaurants) *and filters*, and return them in a fixed structured format.
> **Input:** _“Italian food near me under 200”_

```json 
{
  "dish": ["pizza", "pasta", "risotto"],
  "price": "<200",
  "reason": "popular Italian dishes available nearby within your budget.",
  “distance”: “nearby”
}
```

## **How we trained it**

- Finetuned Gemma fully earlier with custom data
- Then a newer model worked with just few shots

### **Data preparation**

> Queries like *“pahado pe hun, kuch garam mil jaaye”* does not exist in any public dataset → synthetic generation needed

- **Synthetic data - from intents to dishes:**   hand-listed thousands intent & discovery style queries _fever_, _date night_, _comfort food_, _pahado pe hun_, _feeling sick_ → larger teacher model → 20–30 dishes that genuinely satisfy it
>  dense pairing of _how people describe a need_ → _what actually answers it_

- **Synthetic data - reverse queries:** teacher model reverse-engineered plausible customer queries from known data
	- “*Given the dish description ‘Steaming hot khichdi with mild spices’, generate 5 **vague** customer queries that would lead to this dish.”  →* “stomach upset food”, “light dinner”, …
	- *“Given this journey, app open → searched ‘japanese’ → filtered price < 400, rating > 4 -> tapped Ebisu (japanese, asian), generate 5 queries that lead to the tapped entity.”  ->“* japanese food under 400”, …

- **Mining our own search logs (query→click) :**  “piz” → pizza (dish) / Pizza Hut / Domino’s (restaurants) → the **click** that followed tells which reading was right
- **Making it survive real Indian search traffic :** 
	- _Hinglish injection_ : 20–30% of queries code-mixed, e.g. “paneer butter masala ka price kya hai”, “badiya chinese food”
	- _Typos, noise and partial input_ : “Pizaa”, “Briyani”, “chicken B”
	- _Regional abbreviations and names_ : curd rice → _perugu annam_ in Telugu
	- _Multi-attribute composition_ : samples that **stack orthogonal constraints**, \[Cuisine\] + \[Dish\] + \[Price\] + \[Dietary\]. “Veg Chinese under 300 without garlic” → **model learns to satisfy several conditions at once instead of latching onto one**
	- _Filter addition_ : queries augmented with ratings, distance, price bands and ingredient negations (“without onion”), so filters are first-class, not an afterthought.

Convert every example into model’s chat template - system turn (schema + rules), a customer turn (query), and an assistant turn (target JSON)

### **From dishes to building blocks**


> [!NOTE] One decision shaped everything downstream: how we represent the catalogue itself
> Historically, Zomato could only understand dishes that were explicitly registered in the system. But registering every dish, in every variation, across millions of menus, is a losing battle, so we stopped trying. Instead of registering dishes, we broke each dish into a **base category plus metadata**
> 
>  **slow cooked chicken biryani**  →  base dish: biryani,  ingredient: chicken,  attribute: slow cooked

This gave us uniformity on both ends. The LLM now generates a base category together with its filters and attributes, and retrieval can resolve it two ways, through structured filter-mapping when the attributes line up, and a regex-based fallback when they don’t. A long tail of unregistered dishes suddenly became reachable, not because we catalogued them, but because we stopped needing to.

### **Fine-tuning with LoRA**

Teaches the model taxonomy, JSON shape, and the *mode-switching behaviour*

### **Few-shot on a newer generation**

> The LoRA model worked, but it also taught us that, once the data and the output contract were solid, most of the remaining difficulty lived in the _data_, not the model. `Interesting - not really sure what this means`

Newer-generation Gemma model, using few well-chosen few-shot examples sampled from the same training set matched or bettered finetuned model

### **From fine-tuned teacher to prompt-driven student**


> [!NOTE] This is just a fancy way of writing - I feel
> In effect, our LoRA fine-tuned model became the **teacher** for a new **student,** the upgraded Gemma-series model, handing down everything it had learned about our schema and output modes through the examples it generated. Behaviour now lived in the **instruction** rather than **baked into weights** → add, remove or reshape a JSON attribute by editing the instructions every time the taxonomy moved

## **How we taught the retrieval brain meaning (The Encoder Model)**

“hot coffee” and “cold coffee” are nearly identical strings but for Zomato they are almost opposites → **contrastive learning**

### **Contrastive learning with triplets**

Construct a large dataset of **(Anchor, Positive, Negative)** triplets

- **Anchor (customer query) :** “Soothing food for a sore throat”
- **Positive (relevant) :** “Chicken Clear Soup, warm, non-spicy broth, easy on the throat.” (a dish plus a one-line reason)
- **Negative (irrelevant) :** “Spicy Schezwan Fried Rice”, “Cold Ice Cream”

### **Hard-negative mining: where the real learning happens**

We mine two kinds:

- **Lexical-overlap negatives :** For “Veg Pizza” a lexically close hard negative is “Chicken Pizza”
- **Vector-space hard negatives (a boosting loop) :** after a training round → retrieve the model’s own top-k false positives, items it wrongly ranked as relevant → LLM-as-judge / cross-encoder → feed them back as negatives for the next round

## **How we know it’s working**

Manually verified evaluation set of **1,000+ diverse queries**, each labelled along a fixed structure:
\| Query Category \| Input Query \| Ground-Truth JSON \| Critical Entity \|

**Metrics:**

- **JSON Validity Rate :** need parsable JSON 
- **Slot-Level F1 :** precision and recall for each individual key (price, dish, dietary, …) → which attributes the model extracts reliably **AND / OR** which need more data
- **Retrieval quality (Recall@k / MRR) :** for the embedding side

> Team: [Gourav Bansal](https://www.linkedin.com/in/gouravbansal), [Saurabh Kalia](https://in.linkedin.com/in/saurabh-kalia-1560a910), [Akhil Bansal](https://in.linkedin.com/in/akhilbansal21), [Devendra Samatia](https://in.linkedin.com/in/devendrasamatia), [Siddharth Kumar Iyer](https://www.linkedin.com/in/siddharth-kumar-iyer-7182ba7a/), [Divyanshu Sahu](https://in.linkedin.com/in/divyanshu-sahu), [Karan Vaish](https://in.linkedin.com/in/karan-vaish-1b3b56169), [Aniket Patel](https://in.linkedin.com/in/aniket-patel-ap), [Garvit Tak](https://in.linkedin.com/in/garvit-tak), [Surya Shukla](https://in.linkedin.com/in/cerdo), [Garvit Chittora](https://in.linkedin.com/in/garvit-chittora), [Akshat Goyal](https://in.linkedin.com/in/itsme-akshat), [Parth Javiya](https://in.linkedin.com/in/parthjaviya)