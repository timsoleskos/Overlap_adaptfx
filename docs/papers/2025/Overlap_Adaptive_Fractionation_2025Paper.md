# Overlap Adaptive Fractionation 2025 Paper

                       Overlap Guided Adaptive Fractionation
          Yoel Pérez Haas∗1 , Lena Kretzschmar*1 , Bertrand Pouymayou1 , Stephanie
                             Tanadini-Lang1 , and Jan Unkelbach1
1
    Department of Radiation Oncology, University Hospital of Zurich, Zurich, Switzerland

                                             November 5, 2025


                                                   Abstract
           Purpose: Online-adaptive, Magnetic-Resonance-(MR)-guided radiotherapy on a hybrid MR-
       linear accelerators enables stereotactic body radiotherapy (SBRT) of abdominal/pelvic tumors with
       large interfractional motion. However, overlaps between planning target volume (PTV) and dose-
       limiting organs at risk (OARs) often force compromises in PTV-coverage. Overlap-guided adaptive
       fractionation (AF) leverages daily variations in PTV/OAR overlap to improve PTV-coverage by
       administering variable fraction doses based on measured overlap volume. This study aims to assess
       the potential benefits of overlap-guided AF.
           Methods: We analyzed 58 patients with abdominal/pelvic tumors having received 5-fraction MR-
       guided SBRT (>6Gy/fraction), in whom PTV-overlap with at least one dose-limiting OAR (bowel,
       duodenum, stomach) occurred in ≥ 1 fraction. Dose-limiting OARs were constrained to 1cc ≤ 6Gy
       per fraction, rendering overlapping PTV volumes underdosed. AF aims to reduce this underdosage
       by delivering higher doses to the PTV on days with less overlap volume, lower doses on days with
       more. PTV-coverage-gain compared to uniform fractionation was quantified by the area above the
       PTV dose-volume-histogram-curve and expressed in ccGy (1ccGy = 1cc receiving 1Gy more). The
       optimal dose for each fraction was determined through dynamic programming by formulating AF as
       a Markov decision process.
           Results: PTV/OAR overlap volume variation (standard deviation) varied substantially between
       patients (0.02–5.76cc). Algorithm-based calculations showed that 55 of 58 patients benefited in PTV-
       coverage from AF. Mean cohort benefit was 2.93ccGy (range -4.44 (disadvantage) to 22.42ccGy).
       Higher PTV/OAR overlap variation correlated with larger AF benefit.
           Conclusion: Overlap-guided AF for abdominal/pelvic SBRT is a promising strategy to improve
       PTV-coverage without compromising OAR sparing. Since the benefit of AF depends on PTV/OAR
       overlap variation—which is low in many patients—the mean cohort advantage is modest. However,
       well-selected patients with marked PTV/OAR overlap variation derive a relevant dosimetric benefit.
       Prospective studies are needed to evaluate AF feasibility and quantify clinical benefits.


1      Introduction
The recent technological development of a hybrid MR(Magnetic Resonance) - linear accelerator (MR-
linac), i.e. a combination of an MR scanner, a 6 MV linear accelerator and a multileaf collimator for
intensity modulation - makes real-time MR-guided radiotherapy (MRgRT) feasible [1][2][3][4], and enables
the delivery of high-precision stereotactic body radiotherapy (SBRT) to traditionally more challenging
target volumes, e.g. mobile abdominal lesions such as pancreatic cancer, as well as liver or adrenal
metastases [5][6][7][8]. Main advantage of MRgRT is the visualisation of patient anatomy with supe-
rior soft-tissue contrast compared to CT-guided radiotherapy. This enables daily, pre-treatment volume
adaption of gross tumor volume (GTV) and organs at risk (OAR).
                                            "Challenge posed by overlap of PTV with OARs"
A challenge in the application of MR-guided SBRT for abdominal/pelvic target volumes is posed by close-
ness or overlap of the planning target volume (PTV) with adjacent, highly vulnerable OARs, e.g. the
bowel and stomach. This issue, if it arises, cannot be improved by simply adjusting the treatment plan
to the geometry of the day, as is routinely performed in MRgRT [9]. Usually, compromises in PTV dose
coverage, or a reduction of prescription dose are necessary, with both of these options being undesirable
    ∗ Shared First Authors




                                                        1
due to their potentially negative effects on local tumor control.

However, owing to interfraction mobility of both GTV and OARs, the geometry between them does not
remain static over the course of treatment, as demonstrated in Figure1. Adaptive fractionation (AF)
[10][11][12] is one approach to utilize these daily geometric changes to provide better dose coverage of
the PTV without compromising sparing of OARs. Its general concept is to apply more dose to the PTV
on days with more favorable PTV/OAR geometry, and less dose on days with less favorable PTV/OAR
geometry.




                        (a)                                                  (b)

Figure 1: a. MR-guided SBRT fraction 2/5 of patient 3 with a pancreatic target volume. The blue
contour represents the GTV, the red contour the PTV, the green contour the dose-limiting OAR (in this
case the bowel). The volume forcing compromises in dose distribution is the overlap region between PTV
and bowel (yellow shading). b. MR-guided SBRT fraction 5/5 of the same patient. The blue contour
again represents the GTV, the red contour the PTV, the green contour the dose-limiting OAR (bowel),
the yellow shading the overlap region. Note how the overlap region is larger than in fraction 2 due to
interfraction bowel displacement.

AF has previously been explored by our group in the context of MRgRT[13]. In this prior work, day-to-day
variation of the geometry was quantified via a so-called "Sparing Factor" defined as the quotient of the
dose received by the dose-limiting OAR (D1cc) and the dose received by the PTV (D95) in each fraction.
AF was implemented by upscaling the dose distribution for a small sparing factor and downscaling for
a large sparing factor. The Sparing Factor is smaller the larger the distance between the PTV and the
OAR. However, the Sparing Factor is approximately 1 if the PTV and the OAR overlap and the minimum
dose in the PTV equals the maximum dose in the OAR. This limits the applicability of AF for patients
in whom PTV and dose-limiting OAR have direct volume overlap in each fraction - because there is little
variation in the sparing factor even if the overlap volume varies substantially.
Based on these previous observations, we have developed an alternative AF approach for cases, in which
direct overlaps between PTV and dose-limiting OAR(s) are present, by investigating overlap volume as the
essential descriptor of geometric variation and main leverage point in the context of AF. We demonstrate
this new approach to AF using data of patients who have previously received 5-fraction SBRT in the form
of MRgRT at the MR-Linac in our department for abdominal and pelvic target volumes, and whose PTV
shows overlap with at least one dose-limiting hollow OAR - the bowel, the duodenom or the stomach.


2    Patient selection and treatment plans
The data for this analysis stems from all patients with abdominal and pelvic primary or secondary tu-
mors, who received SBRT at the MR-linac (MRIdian, ViewRay) in our department between 2019 - 2024.
Patients were included in the analysis, if the PTV was located in direct proximity to the dose-limiting
OARs bowel, stomach, and/or duodenum, overlapping with at least one of these organs in at least one
fraction of treatment, and thus necessitating a PTV coverage compromise to respect OAR constraints.
Additionally, to qualify for inclusion, SBRT had to have been administered in 5 fractions with a minimum
dose of 6 Gy per fraction, regardless of the prescribed isodose level. To limit complexity, in this pilot



                                                     2
investigation treatment courses were excluded, if more than one lesion was irradiated in the same plan.

We isolated a total of 58 treatment courses that met these criteria and extracted volumetric and dosimet-
ric information on GTV, PTV and relevant OARs for a total of 6 treatment plans per course (the initial
plan based on the planning MRI, plus the 5 delivered plans) from our TPS. Treatment planning and de-
livery for all included patients followed institutional protocols: Each patient underwent either MR-only, how????
or combined MR- and CT-simulation, as well as daily MR imaging for online adaptive radiotherapy. Tu-
mors and OARs within a 2 cm margin around the GTV were recontoured prior to each fraction according
to institutional guidelines. Adaptive treatment plans were generated daily to account for inter-fractional
anatomical changes, with prescription doses remaining constant across all fractions.

Overlap between PTV and the relevant, dose-limiting OARs was calculated by creating an overlap struc-
ture for each overlapping OAR with the PTV and measuring its volume. If more than one dose-limiting
OAR showed overlap, all separate overlap volumes were added together to create a total overlap volume.
This was possible due to the fact that bowel, duodenum and stomach have the same OAR objectives
applied to them in our institutional protocol. All considerations on AF below were made based on this
total overlap volume.

The most common malignancies in the cohort were primary and secondary tumors of the pancreas (n=24),
followed by adrenal metastases (n=13) and abdominal/pelvic lymph node metastases (n=13); the remain-
ing patients had abdominal tumor manifestations that were not more closely defined (n=8), e.g. non-
lymph-node-associated soft-tissue growths. 32 patients had 35 Gy prescribed to the PTV as total dose,
14 received 33 Gy, 11 received 40 Gy, and one patient was prescribed 45 Gy. Due to the daily-adaptive
nature of the workflow, PTV volumes showed fluctuations over the course of radiotherapy, depending on
the daily size of the GTV, and influenced by other factors such as interobserver variety between different
contouring physicians and the image quality of the daily MRI scan. PTV volume interfraction variation,
i.e. standard deviation, was generally <10% of PTV volume; only 7 patients showed a variation of PTV
size of >10% over the course of SBRT, with a maximum of 18% PTV volume in a single patient due to
tumor swelling and reduced visibility conditions creating uncertainties in the adaptive planning MRIs.

Mean PTV volumes per course were varied and ranged from 1.51 cc to 286 cc in the patient cohort. No
clear dependency between PTV volume and tumor entities/locations was found in the patient cohort,
with all tumor categories (pancreatic, adrenal, lymph nodes, unspecified abdominal lesions) containing
both small and large PTV volumes.


2.1    Overlap Variation
PTV/OAR overlap volume variation, i.e. its standard deviation, showed a high variability between
patients, ranging from 0.02 cc up to 5.76 cc. Figure 2 shows the ten patients with the largest overlap
variations in the cohort and their respective PTV/OAR overlaps, stratified by fraction; a detailed chart
of the overlaps of all patients can be found in the Appendix in Figure 15.




                                                    3
Figure 2: Variation in overlap volumes for the ten patients with the highest standard deviation, sorted by
magnitude of variation (highest to lowest). Each point represents the total calculated overlap volume for
one fraction, including the initial simulation. The red line represents the mean overlap volume throughout
each treatment course.

Figure 3 shows the distribution of the patient-specific standard deviations for all 58 patients. We fitted
a gamma distribution to this empirical data, which will later serve as a population-level prior in the AF
framework introduced in Section 3.2.




Figure 3: Gamma prior fitted to the observed overlap volume standard deviations of all patients, resulting
in a shape parameter k = 1.07 and a scale parameter θ = 0.78. The prior distribution (solid line) has
been rescaled to match the histogram of the observed standard deviations (histogram).

Both in absolute and in relative numbers (=calculated as part of PTV volume) we found a statistically
significant correlation between mean PTV/OAR overlap over the course of treatment and the variation
(standard deviation) of PTV/OAR overlap, as illustrated in Figure 4. Overlap variation did not show a
relevant dependency on mean PTV volume or tumor entity.




                                                    4
                         (a)                                                  (b)

Figure 4: a. Correlation between absolute mean PTV/OAR overlap volume and its variation (standard
deviation), calculated per treatment course. b. Correlation between relative PTV/OAR overlap and
relative standard deviation of overlap in relation to PTV-volume. The colors in both plots represent the
respective tumor locations.

Looking at the relative PTV/OAR overlaps and their respective Standard Deviations we found an in-
teresting trend of small target volumes (especially lymph nodes) having a - for their size - high relative
overlap variation with dose-limiting OARs, despite absolute overlap volumes and variations being gener-
ally small in these patients. This led us to further investigate the potential benefits of AF not only in
absolute but in relative numbers as well. Results of this analysis are shown in Chapter5.



3     Adaptive Fractionation
AF aims to adjust the dose delivered in each treatment fraction based on daily anatomical variations,
in this case, the overlap between the PTV and OARs. The key idea is to escalate the dose when the
anatomy is favorable (small overlap) and reduce the dose when there is significant overlap.

Treatment planning is subject to the following clinical objectives and constraints:

    1. The dose in the OAR (including the overlap volume) is constrained to 6 Gy in each fraction. In
       this study, we consider hypofractionated treatments consisting of F = 5 fractions. Thus, this
       corresponds to a commonly used cumulative dose constraint of 30 Gy in 5 fractions.
    2. The total cumulative dose prescribed to the PTV is larger than 30 Gy but the OAR constraint is
       prioritized over PTV coverage. Consequently, the PTV is underdosed in the overlap region.

    3. The per-fraction dose prescribed to the PTV, dt , must lie within the the range dt ∈ [dmin , dmax ],
       where dmin is set equal to the OAR constraint of 6 Gy.
The goal is to determine the daily prescription doses dt for each fraction t such that the overall PTV
underdosage is minimized. To that end the following quantitative measure of PTV underdosage is intro-
duced.

3.1     Measure of Underdosage
To characterize inter-fractional anatomical variability, we quantify the volumetric overlap between the
PTV and OARs. Specifically, let ot denote the PTV volume overlapping with the relevant OAR on
treatment day t measured in cc. As an approximate measure for the loss in PTV coverage in fraction t,
we introduce a cost function ct defined as

                                            ct = ot · (dt − dmin )                                      (1)
Here, dt is the prescribed dose to the PTV in fraction t and dmin is the dose that is deliverable to the
overlapping region without violating the per-fraction OAR constraint. The cost ct is thus proportional
to the amount by which the prescribed fraction dose exceeds dmin .



                                                      5
A schematic representation of this concept is shown in Figure 5a. The illustration schematically shows
the dose-volume histogram (DVH) of the PTV for a prescription dose dt > dmin . The green area above
the DVH curve represents the underdosage of the PTV due to the dose constraint in the overlap volume.
The red box represents the approximation of the area above the DVH curve as defined by the metric ct .
Thus, the cost is measured in units of ccGy, where an improvement of 1ccGy indicates that 1cc of the
PTV volume receives 1 Gy more. The green area can only be determined after a treatment plan has been
created for a prescription dose dt whereas the red area represents an approximation thereof that can be
estimated based on the overlap volume ot alone prior to treatment planning.




(a) Schematic illustration of the measure of PTV under-           (b) Comparison of DVHs for different dose pre-
dosage. The blue line represents the PTV DVH. The                 scriptions and overlap scenarios. The green line
dashed red line indicates the overlap volume. The red box         shows a minimum dose (dmin ) resulting in no un-
corresponds to the cost ct , while the green shaded area in-      derdosage regardless of overlap. The blue line cor-
dicates the actual PTV underdosage due to the PTV–OAR             responds to a uniform dose (dref ) delivered under
overlap.                                                          high-overlap conditions, resulting in significant un-
                                                                  derdosage (blue box). The orange line represents a
                                                                  high dose delivered on a low-overlap day, yielding
                                                                  moderate underdosage despite the higher dose (or-
                                                                  ange box). Dashed red lines mark overlap volumes.


The overall quality of an adaptive fractionation scheme can be evaluated by aggregating the daily costs
over the entire treatment course. We define the total accumulated cost as:
                                                      F
                                                      X
                                           ctotal =         ot (dt − dmin )                                        (2)
                                                      t=1

To illustrate the rationale for AF based on overlap further, Figure 5b illustrates two fractions with
different overlap volumes. Suppose the overlap volume is o1 = 10 cc in the first fraction and o2 = 3 cc
in the second fraction. Let us further assume, a uniform fractionation scheme prescribes a constant dose
of dref = 8 Gy in both fractions, with an OAR constraint of dmin = 6 Gy. The blue line illustrates the
treatment plan on the day with high overlap when a dose of dref = 8 Gy is prescribed, corresponding to
an approximate cost of 10 cc · (8 − 6) Gy as illustrated by the blue-shaded box. The total cost over two
fractions for uniform fractionation is


             Uniform Fractionation: d1 = d2 = 8 Gy
                                         ctotal = 10 cc · (8 − 6) Gy + 3 cc · (8 − 6) Gy = 26 ccGy


In contrast, an adaptive strategy could deliver 6 Gy in the first (unfavorable) fraction and 10 Gy in the
second (favorable) fraction. Since the dose of 6 Gy respects the OAR constraint, no underdosage occurs
in fraction one regardless of the overlap volume, resulting in zero cost (green line). The orange line shows
the DVH for the high-dose prescription of 10 Gy for the second fraction with a small overlap, resulting
in a cost of 3 cc · (10 − 6) Gy as illustrated by the orange-shaded box. The total cost over two fractions
for adaptive fractionation is


            Adaptive Fractionation:      d1 = 6 Gy,     d2 = 10 Gy
                                         ctotal = 10 cc · (6 − 6) Gy + 3 cc · (10 − 6) Gy = 12 ccGy



                                                            6
In this simplified example, adaptive fractionation reduces the cost by 14 ccGy compared to uniform
fractionation. Even though the underdosage in fraction two increases when 10 Gy instead of 8 Gy is
prescribed, this is overcompensated for by avoiding the substantial underdosage in fraction one that
would otherwise occur due to the large overlap volume.

3.2    Probability distribution for overlap volumes
The central challenge in AF is that future anatomical configurations are not known in advance. Adaptive
fractionation relies on anticipated overlap volumes for future fractions for determining the optimal dose
in the current fraction. In this work, we assume that the daily overlaps follow a normal distribution
ot ∼ N (µ, σ 2 ), which is updated iteratively during treatment. As in [13], we estimate the patient-specific
parameters µ and σ based on observed overlap values up to the current fraction.
Because mean overlap volumes can differ significantly across patients, we use the maximum likelihood
estimator to compute the patient-specific mean, updating it with each observed value:
                                                              t
                                                        1 X
                                               µt =              oτ                                      (3)
                                                      t + 1 τ =0
Here, o0 denotes the overlap volume observed in the planning MR.

The standard deviation estimated from the observed overlap volumes up to fraction t is given by
                                           v
                                           u
                                           u 1 X     t
                                     pat                          2
                                    σt = t              (oτ − µt )                                       (4)
                                             t + 1 τ =0

Since the standard deviation strongly influences the adaptive fractionation policy, we adopt a Bayesian
approach to regularize early estimates. Specifically, we use a population-based gamma prior for σ:
                                                                     
                                                   1               −σ
                                  f (σ; k, θ) =        σ k−1
                                                             exp                                    (5)
                                                Γ(k)θk              θ
The hyperparameters k (shape) and θ (scale) were estimated empirically from the distribution of standard
deviations across our retrospective cohort of 58 patients (Figure 3), yielding k = 1.07 and θ = 0.78.

Given this prior and the observed overlaps up to fraction t, we compute the maximum a posteriori
estimate of σt . The likelihood is based on a normal model with unknown variance, and the resulting
maximum a posteriori estimate can be obtained numerically by maximizing the posterior:
                                          "                              !#
                                            σ k−1              (σtpat )2
                             σt = argmax          exp(−θσ) exp       2                          (6)
                                     σ      σ t−1                2 σt
The a posteriori estimator balances patient-specific observations with population-informed regularization,
yielding more robust estimates, especially in the early treatment phase when few overlap volumes were
observed.

3.3    Markov Decision Process Model
To solve the adaptive fractionation problem under uncertainty about future overlaps, we model the de-
cision process as a Markov Decision Process (MDP) similarly to [13]. The goal is to find a dose selection
policy that minimizes the expected cumulative cost over F treatment fractions, while delivering the pre-
scribed cumulative dose to the PTV.

The MDP is described by states, actions, state transitions and cost, which are given by:

State.
In each fraction t, the state is defined as a tuple st = (ot , Dt ) where:
   • ot is the observed PTV–OAR overlap volume in fraction t,
            Pt−1
   • Dt = τ dτ is the cumulative dose delivered to the PTV up to (but not including) fraction t.


                                                        7
This state encapsulates the current geometric condition and treatment history, both of which influence
the optimal dose decision.

Action.
The action is the dose dt prescribed to the PTV in fraction t. The action space is bounded:

                                                        dt ∈ [dmin , dmax ]                                (7)

In this work the lower bound dmin is set to 6 Gy, equivalent to the dose constraints in the OAR and the
upper bound is 10 Gy.

State Transition.
After applying dose dt in state st = (ot , Dt ), the state transitions to a new state st+1 = (ot+1 , Dt+1 ),
where:
   • ot+1 ∼ N (µt , σt2 ) is drawn from the patient-specific normal distribution,

   • Dt+1 = Dt + dt is the updated cumulative PTV dose.
Since ot+1 is stochastic, the transition is probabilistic.

Cost.
The immediate cost in each fraction is defined by the formerly introduced cost:

                                                    ct = ot · (dt − dmin )                                 (8)

In addition, we impose a terminal cost cF +1 to ensure that the total prescribed PTV dose Dpres is
delivered exactly:
                                        (
                                           0 if DF + dF ≥ Dpres
                                cF +1 =                                                        (9)
                                           ∞ otherwise

3.4    Dynamic Programming Algorithm.
The optimal policy πt (o, D) assigns to each state (o, D) the optimal action that minimizes the expected
cumulative cost until the end of treatment in fraction F . To compute the optimal policy for adaptive
fractionation, we apply dynamic programming (DP), which recursively computes the optimal dose to
deliver in each fraction by minimizing the sum of immediate cost and expected future cost [14]. DP is
based on the value function vt (o, D), which quantifies the expected future cost when starting from state
(o, D) in fraction t and following the optimal policy from there on. Optimal policy and value function can
then be calculated in a backward recursion starting in the last fraction t = F and iterating backwards to
t = 1. One iteration of DP is given by
                                            "                                          #
                                                               X
                      vt (o, D) =    min     o · (d − dmin ) +       ′        ′
                                                                 P (o )vt+1 (o , D + d)               (10)
                                  d∈[dmin ,dmax ]
                                                                            o′
                                                    "                                                 #
                                                                            X
                    πt (o, D) =     argmin              o · (d − dmin ) +            ′       ′
                                                                                 P (o )vt+1 (o , D + d)   (11)
                                  d∈[dmin ,dmax ]                           o′

where the value function is initialized through the terminal cost, i.e. vF +1 (o, D) = cF +1 (o, D). Here,
P (o′ ) denotes the probability distribution over the unknown overlap volume in the next fraction, which is
given by the normal distribution N (µ, σ 2 ). The sum over o′ accounts for the stochastic nature of future
anatomy.

To compute vt and πt in practice, we discretize both the state space, i.e. D and o, and the action space
[dmin , dmax ]. Note that we make the approximation that the estimates of µ and σ are not part of the
state of the MDP but are assumed to be constant over future fractions. To apply AF to a given patient
with a given sequence of overlap volumes, we update µ and σ in each fraction as described in section 3.2
and recompute the optimal policy.



                                                                8
3.5    Benefit Evaluation
To assessPthe benefit of AF, we first evaluate fractionation schemes based on the cumulative cost
            F
ctotal = t=1 ot (dt − dmin ) as measure of PTV underdosage. In a second step, replanning of selected
patients is performed.

1. Benchmarking based on cost.
We compare ctotal for AF to two benchmarks:

    • Uniformly Fractionated Treatment: This corresponds to the standard treatment as delivered in
      practice, where the dose is uniformly fractionated across all days. The total accumulated cost ctotal
      for this treatment serves as a baseline for comparison.
    • Upper Bound: This represents an idealized scenario in which the full sequence of overlap volumes
      ot is known in advance. Based on this, we compute the optimal set of fraction doses that minimize
      the total cost ctotal . This plan serves as an upper bound on what any realistic adaptive policy could
      achieve.

2. Evaluation through replanning.
In a second step, we assess the benefit of AF based on the actually achievable PTV-DVHs. To that
end, replanning of all fractions is performed in the MR-Linac treatment planning system for selected
patients, using the prescribed doses per fraction recommended by the AF algorithm. For each replanned
fraction, we extract the dose–volume histogram (DVH) and compute the underdosage in the PTV via the
area above the DVH curve (as illustrated in Figure 5a). A complete treatment course is characterized by
summing the areas above the DVH across all fractions. By comparing these replanned AF-based fractions
with the clinically delivered ones, we can quantify the actually realized dosimetric benefit of AF.


4     Illustration of Adaptive Fractionation
To illustrate the behavior of the AF algorithm, we present two patients in detail for whom we characterize
the optimal policy and discuss the AF scheme and its benefit.


4.1    Patient 3 (33 Gy Prescription)
Patient 3 was prescribed a total dose of 33 Gy in 5 fractions. Given that 30 Gy are delivered by applying
the minimum dose in each fraction, the prescribed PTV dose exceeds this minimum dose by only 3 Gy.
Consequently, the maximum dose that can be delivered in any fraction is 9 Gy. The PTV–OAR overlap
volumes were o0 = 9.08 cc on the planning MRI, followed by [19.79, 6.02, 9.45, 19.59, 12.62] cc across
the five treatment fractions.

At the start of the first fraction, the estimated mean and standard deviation of the probability distribution
over future overlap volumes are estimated from the planning scan and the first fraction scan. This results
in µ1 = 14.4 cc and σ1 = 3.2 cc, respectively, as computed via Equation 6. The resulting overlap
distribution is shown in orange in Figures 6a and b.




                                                     9
                         (a)                                                    (b)




                         (c)                                                    (d)




                         (e)                                                    (f)

Figure 6: Optimal policies and value functions for patient 3. a,c,d,e and f show the optimal policies for
fractions one through five. For the first fraction the probability distribution over the overlaps is shown in
orange. For the remaining fractions it’s shown in white. The probability density values are not shown for
better readability. b is the expected cost for first fraction. The black crosses mark the observed overlap
and the resulting dose policy/value.

Figure 6a shows the optimal policy (blue curve) computed using the current probability distribution
(orange). Since no dose has yet been delivered, the full range of allowed doses (6–9 Gy) is available.
Interestingly, the algorithm yields a binary policy: the maximum dose (9 Gy) is prescribed if the overlap
is less than 11.7 cc, and the minimum dose (6 Gy) is prescribed if the overlap exceeds 11.8 cc. This
threshold lies at a cumulative probability of roughly 20%, i.e. the dose of 9 Gy is delivered if the prob-
ability of observing a lower overlap volume in the future is less than 20%. The binary nature of the
optimal policy originates from the immediate cost being a linear function of the prescribed dose. As a
consequence, the optimal dose to deliver is either the minimum dose or the maximum dose.

Figure 6b shows the corresponding value function in fraction one. The value function is constant for
overlaps above 11.7 cc where 6 Gy is delivered in the first fraction. In this case, the immediate cost in
fraction 1 is zero and the value function is given by the expected future cost from fraction two onward.


                                                     10
For overlaps below 11.7 cc where 9 Gy is delivered in the first fraction, the value function is given by
the immediate cost in fraction one and the expected future cost from fraction two onward is zero. The
value function is therefore linear in the overlap volume. The threshold of 11.7 cc is overlap where the
immediate cost of delivering 9 Gy in the first fraction equals the expected future cost from fraction two
onward if 6 Gy is delivered in the first fraction.

In the second fraction (Figure 6c), a favorable overlap of o2 = 6.02 cc is observed. The updated distribu-
tion has a lower mean (µ2 = 11.63 cc) and a slightly higher standard deviation (σ2 = 3.88 cc). Again, the
policy exhibits a sharp decision boundary. For overlap volumes larger than 9.2 cc, the algorithm suggests
to deliver the minimum dose of 6 Gy irrespective of the dose delivered in the first fraction. For overlap
volumes smaller than 9.2 cc, corresponding to the 27th percentile, the policy depends on the previously
accumulated dose in the PTV: if only 6 Gy was delivered previously, the algorithm chooses to deliver 9
Gy now. However, if 9 Gy had already been delivered earlier, the policy prescribes only 6 Gy to avoid
exceeding the cumulative PTV dose prescription.

The optimal policy in fractions three (Figure 6e) and four (Figure 6f) is qualitatively analogous to frac-
tion two. The policy exhibits a sharp decision boundary. For overlap volumes above a threshold, the
minimum dose of 6 Gy is delivered; for overlap volumes below the threshold, the maximum dose to reach
to total prescribed PTV dose is delivered. However, threshold shifts to larger percentiles of the overlap
distribution for later fractions. In fraction four, the decision boundary lies at the 50th percentile, which is
an intuitive result. If there is a probability larger than 50% that the observed overlap in the last fraction
is lower than the overlap observed in the fourth fraction, 6 Gy will be delivered in the fourth fraction.
Otherwise, 9 Gy will be delivered in the fourth fraction.

Finally, in the fifth fraction (Figure 6g), no further adaptation is possible. The remaining dose required
to meet the total PTV prescription is delivered, regardless of the observed overlap.

We summarize the results of the adaptive fractionation (AF) treatment for Patient 3 in Table 1, alongside
two benchmarks: the upper bound and the uniformly fractionated reference treatment. The upper bound
corresponds to a treatment that delivers 9 Gy in the fraction with the smallest overlap. Although this
is not known in advance, the AF algorithm takes the right decision for this patient. For the sequence of
overlap volumes for this patient, the AF algorithm identifies the overlap volume in fraction two as small
compared to what is expected based on planning scan and fraction one, and delivers the maximum dose
of 9 Gy accordingly. Since indeed no smaller overlap volumes are observed in fractions three to five, AF
achieves the lowest possible PTV underdosage given by the upper bound.

In contrast, the reference treatment follows uniform fractionation and applies the same dose (6.6 Gy) in
every fraction, regardless of daily anatomical variations. As a result, the total cost accumulated is 40.5
ccGy, compared to just 18 ccGy for the AF plan. This translates to a reduction in cost of 22.5 ccGy,
representing a better coverage of the PTV. For context, the average PTV volume patient 3 is 69.5 cc,
with a mean overlap of 12.7 cc. This means that approximately 20% of the PTV was overlapping with the
OAR, and would effectively receive an additional 2 Gy with adaptive fractionation compared to uniform
fractionation.



     Fraction        Overlap     Upper Bound      Adaptive Fractionation           Reference Plan
                                  Dose [Gy]       Dose [Gy] Cost [ccGy]        Dose [Gy] Cost [ccGy]
  Planning MR        9.08 cc
  First Fraction     19.79 cc          6               6            0              6.6           11.8
 Second Fraction     6.02 cc           9               9          18.06            6.6            3.6
 Third Fraction      9.45 cc           6               6            0              6.6            5.7
 Fourth Fraction     19.59 cc          6               6            0              6.6           11.8
  Fifth Fraction     12.62 cc          6               6            0              6.6            7.6
   Total Cost                                              18 ccGy                       40.5 ccGy


                       Table 1: Dose delivered to the PTV for different treatments.



                                                      11
4.2    Patient 49 (40 Gy Prescription)
Patient 49 was prescribed a total dose of 40 Gy. The observed PTV–OAR overlaps were 4.27 in the plan-
ning scan and [5.91, 3.28, 3.87, 4.18, 8.36] cc across the five treatment fractions. The allowed dose range
spans from 6 Gy to 10 Gy per fraction. Unlike Patient 3, where, due to the low prescription dose, only
one high-dose fraction was delivered and the optimal policy was described by a binary decision boundary,
Patient 49 requires several high-dose fractions.

Figure 7 shows the optimal policy over the five treatment fractions. AF leads to a strategy where two
fractions deliver the maximum dose of 10 Gy, two fractions the minimum of 6 Gy. In addition, one
fraction of 8 Gy is delivered to achieve the 40 Gy prescription dose. Consequently, the optimal policy is
no longer characterized by a single threshold, but rather multiple decision boundaries that partition the
probability distribution of future overlaps.

In the first fraction (Figure 7a), the overlap space is divided into three regions: the algorithm prescribes
10 Gy for overlaps below the 40th percentile, 8 Gy for overlaps between the 40th and 60th percentiles,
and 6 Gy otherwise. As the observed overlap on this day was 5.91 cc, above the 60th percentile, the
algorithm selects the minimum dose of 6 Gy.

In the second fraction (Figure 7c), the policy shows three decision boundaries along the overlap axis,
leading to four distinct regions.
   • Below the 26th percentile:
     A dose of 10 Gy is delivered irrespective of the previously delivered dose.

   • Between the 26th and 50th percentile:
     If 10 Gy was delivered in the first fraction, a dose of 8 Gy is delivered.
     If 8 Gy or less was delivered in the first fraction, overlaps in this range trigger a 10 Gy prescription.
   • Between the 50th and 74th percentile:
     If 8 Gy of more was delivered in the first fraction, a dose of 6 Gy is delivered.
     If 6 Gy was delivered in the first fraction, 8 Gy is prescribed.
   • Above the 74th percentile:
     A dose of 6 Gy is delivered irrespective of the previously delivered dose.
In this fraction, a favorable overlap of 3.28cc is observed, leading the algorithm to prescribe 10 Gy.

The third fraction (Figure 7e) exhibits two decision boundaries and three regions.
   • Below the 35th percentile:
     If the patient has already received 20 Gy in total, the algorithm cannot prescribe another 10 Gy
     and prescribes 8 Gy.
     If the patient has received 18 Gy or less, 10 Gy is prescribed.
   • Between the 35th and 65th percentile:
     If the patient received 18 Gy or more, 6 Gy is prescribed.
     If the patient received 16 Gy, 8 Gy is prescribed.
     If the patient received 14 Gy or less, 10 Gy is prescribed.

   • Above the 65th percentile:
     If the patient has received only 12 Gy in total, the algorithm cannot allow another 6 Gy fraction
     and prescribes 8 Gy.
     If the patient has received 14 Gy or more, 6 Gy is prescribed
For the observed overlap for this patient, we assign 10 Gy since the overlap is slightly below the 49th
percentile.

By the fourth fraction (Figure 7f), only two doses remain to be allocated. Consequently, the policy
simplifies to a binary threshold at the 50th percentile: the larger dose level is delivered if the observed
overlap is below the mean, and the lower dose level otherwise. Since the observed overlap is 4.18 cc,
favorable relative to the updated distribution, the algorithm selects to deliver the larger dose of 8 Gy. In


                                                     12
the fifth and final fraction, no further adaptation is possible and the residual dose is delivered to match
the prescription, for this patient 6 Gy.




                        (a)                                                   (b)




                        (c)                                                   (d)




                        (e)                                                   (f)

Figure 7: Optimal policies and values for patient 49. a,c,d,e and f show the optimal policy for fractions
one through five. For the first fraction the probability distribution over the overlaps is shown in orange.
For the remaining fractions it’s shown in white. The probability density values are not shown for better
readability. b is the expected cost for first fraction. The black crosses mark the observed overlap and the
resulting dose policy/value.

We summarize the AF scheme and the resulting dosimetric cost for Patient 49 in Table 2, comparing
AF to the upper bound and the uniformly fractionated reference treatment. The AF plan successfully
exploits the small overlap volumes of 3.28 cc and 3.87 cc in the second and third fraction by delivering a
high dose of 10 Gy. It further avoids delivering high doses during unfavorable configurations. Both the
first and last fraction had overlap values above average and received only 6 Gy, minimizing the cost. The
8 Gy fraction is delivered at the intermediate overlap of 4.18 cc in fraction four. Overall, the adaptive
strategy achieved a total cost of 37 ccGy, 14.2 ccGy lower than the 51.2 ccGy incurred by the uniform
8Gy-per-fraction reference treatment. Thereby, AF realizes the optimal treatment given by the upper
bound.


                                                    13
     Fraction        Overlap    Upper Bound       Adaptive Fractionation               Reference Plan
                                 Dose [Gy]        Dose [Gy] Cost [ccGy]            Dose [Gy] Cost [ccGy]
  Planning MR         4.27 cc
  First Fraction      5.91 cc         6                 6                 0             8           11.8
 Second Fraction      3.28 cc         10                10              13.1            8            6.6
 Third Fraction       3.87 cc         10                10              15.5            8            7.7
 Fourth Fraction      4.18 cc         8                 8                8.4            8            8.4
  Fifth Fraction      8.36 cc         6                 6                 0             8           16.7
   Total Cost                                                 37 ccGy                       51.2 ccGy


                       Table 2: Dose delivered to the PTV for different treatments.


5    Evaluation Across All Patients
We applied the AF algorithm to all 58 patients in the dataset. Since the immediate cost is linear in the
prescribed dose, the AF tends to deliver either the minimum or the maximum dose. An intermediate
dose level is delivered only once of needed to match the cumulative prescription. Consequently, for a
given cumulative prescription, AF yields the same fractionation scheme and only the oder in which the
fractions are delivered depends on the observed overlap volumes. For the cumulative prescription levels
used in these patients, the algorithm resulted in the following fractionation schemes.

    • For the 33 Gy prescription, the AF plan assigns one 9 Gy fraction and four 6 Gy fractions.
    • For the 35 Gy prescription, the plan includes one 10 Gy, one 7 Gy, and three 6 Gy fractions.
    • The 40 Gy prescription is distributed as two 10 Gy, one 8 Gy, and two 6 Gy fractions.
    • The 45 Gy prescription consists of three 10 Gy, one 9 Gy, and one 6 Gy fraction.

Figure 8a provides an overview of the benefit of AF compared to uniform fractionation across all patients.
Most patients show only a small benefit of adaptive fractionation. The mean benefit was 2.9 ccGy.
However, selected patients show a substantial benefit. The maximum benefit was achieved for Patient
3 (22.5 ccGy). For three patients, AF resulted in worse PTV coverage than uniform fractionation.
The largest reduction in PTV coverage was -4.4 ccGy, In Figure 8b, the benefit is plotted against the
standard deviation of the observed overlap volumes. As expected, a clear trend emerges: patients with
higher variability in PTV–OAR overlap tend to benefit more from AF. The full list of AF treatments for
all patients is provided in the Appendix in table 8.




(a) Histogram with observed benefits and average ben-        (b) Correlation between standard deviation of overlap
efit after applying the AF algorithm.                        volume per treatment course, prescription dose and the
                                                             associated PTV-coverage benefit with adaptive frac-
                                                             tionation. The three patients showing negative benefits
                                                             (<0 ccGy) had a disadvantage from adaptive fraction-
                                                             ation.

Figure 8: Comparison of benefit distribution and standard deviation correlation for adaptive fractionation.


                                                        14
In table 3 we summarize the three patients with the largest improvements after patients 3 and 49 discussed
in the previous section. For patients 14 and 55 the algorithm correctly identified the lowest overlaps and
delivered the high dose levels, realizing the optimal fractionation scheme. In patient 7 the two fractions
with the highest overlap were correctly detected and 6 Gy was prescribed. However, ideally a 10 Gy
fraction was delivered in the last fraction rather than the third fraction, which would have resulted in
9.26 ccGy benefit instead of 7.5 ccGy.

      Fraction              Patient 14                  Patient 55                  Patient 7
                     Overlap [cc] Dose [Gy]      Overlap [cc] Dose [Gy]     Overlap [cc] Dose [Gy]
         0               2.88                        3.00                        5.41
         1               3.35           6            6.41           6            8.20           6
         2               3.35           6            4.13          10            9.16           6
         3               0.53          10            4.91           8            8.18          10
         4               1.92           7            5.66           6            5.45          10
         5               3.66           6            3.56          10            7.18           8
      Benefit                8.8 ccGy                     8.8 ccGy                   7.5 ccGy
    Explanation      Correctly exploited the     Correctly exploited the    Correctly identified the
                     lowest overlap in frac-     lowest overlap in frac-    largest overlap in frac-
                     tions 3 and 4               tions 2, 3 and 5           tions 1 and 2, but de-
                                                                            livered the high doses in
                                                                            suboptimal order.

    Table 3: Observed overlaps and delivered doses for patients where AF improved PTV coverage.

Table 4 analyzes the three patients where AF resulted in worse PTV coverage compared to uniform
fractionation.
   • Patient 16 had a very small overlap in the planning scan, leading the algorithm to expect similarly
     favorable geometry in later fractions. As a result, it saved dose for later, but an unusually high
     overlap in the final fraction led to an overall worse treatment.

   • Patient 5 also had the smallest overlap in the planning scans. The algorithm delayed the delivery
     of the high-dose fraction until the last fraction, which had a relatively high overlap.
   • Patient 32 presented a very large overlap after the planning scan. The algorithm delivered a large
     dose in the first and second fraction when the overlap was smaller. However, an even better overlap
     occurred in the last fraction, which could no longer be exploited.

These cases illustrate the limitations of the AF. Especially in cases where early observations are not
representative for the true underlying distribution of overlap volumes, the algorithm may take decisions
that, in hindsight, are suboptimal.




                                                   15
      Fraction              Patient 16                   Patient 5                   Patient 32
                     Overlap [cc] Dose [Gy]      Overlap [cc] Dose [Gy]       Overlap [cc] Dose [Gy]
         0               1.38                        17.15                        1.56
         1               3.64            6           18.68           6            0.48           10
         2               6.73            6           18.35           6            0.22           10
         3               3.42           10           28.56           6            0.19            8
         4               5.99            8           21.37           6            0.33            6
         5               9.17           10           22.98           9            0.00            6
      Benefit                -4.4 ccGy                   -3.0 ccGy                    -0.7 ccGy
    Explanation      Small overlap in the        Small overlap in the         Large overlap in the
                     planning scan not repre-    planning scan not repre-     planning scan not repre-
                     sentative for later frac-   sentative for later frac-    sentative for later frac-
                     tions.    A large dose      tions.                       tions. Smallest overlap
                     had to be delivered in                                   in the last fraction was
                     the last fraction de-                                    not exploited.
                     spite showing the largest
                     overlap.

Table 4: Observed overlaps and delivered doses for patients where adaptive fractionation performed worse
than uniform fractionation.

In Figure 9, we compare the benefit of AF to the theoretical upper bound that could be achieved if the
overlap volumes of all fractions were known in advance. Out of 58 patients, AF achieved the same ben-
efit as the upper bound in 28 patients. The remaining 30 patients showed varying degrees of deviation.
Figure 9 shows the distribution of differences in benefit between AF and the upper bound for these 30
patients. On average, the benefit of AF is 2.4 ccGy below the upper bound.

Several patients show notable discrepancies. For instance, patient 16 falls short by 22.1 ccGy. The patient
shows substantial variation in overlap volumes and, in theory, could have benefited substantially from
adaptive fractionation. However, because the small and large overlaps are observed in an unfavorable
order, AF actually led to a worse treatment compared to uniform fractionation. Similarly, patient 5
falls short 13.9 ccGy due to a suboptimal sequence of overlaps. Patient 4 also showed a benefit of
approximately 1 ccGy but the benefit could have been 8.0 ccGy higher if the high dose fraction had been
delivered in the best fraction (see Appendix, table 8).




Figure 9: Histogram with AF benefits vs the upper bound. Patients 16, 4 and 4 have been labeled. Only
patients are shown where AF did not deliver the theoretically optimal treatment.

Taken together, these results indicate that selected patients with substantial overlap variation (such as
3, 49, 14, 55 and 7) may benefit from AF and the approach even matches the theoretical upper bound on
the benefit. On the other hand, in few cases AF may yields worse PTV coverage compared to uniform
fractionation. This overlap variations occur in an unfavorable order, especially if the overlap observed in
the planning scan is an outlier. For a large group of patients with little anatomical variation, the benefit
of AF is modest.




                                                    16
In Chapter2.1 we noted our observation of smaller tumors (particularly lymph nodes) showing a high
relative overlap variation despite small absolute overlap volumes. Following up on this observation, we
also investigated the potential benefits of AF in relative numbers. We observed that apart from patients
like 3, 49 and 14, who show large absolute benefits from AF as illustrated in Chapter5, relevant relative
benefits from AF can also be found in patients with smaller PTV and PTV/OAR overlap volumes; for
example in Patients 20, 24 and 26: These three patients all have a lymph node target and a mean PTV
volume of approximately 10 cc, an overlap variation of <1 cc (0.7 - 0.9), and achieve an absolute benefit
from AF of 4.05, 3.97, and 5.69 ccGy each. Calculated in numbers relative to their mean PTV and
PTV/OAR overlap volumes, this represents a sizable benefit in PTV coverage of roughly 2 Gy within
the overlap volume, which makes up 20% of their total PTV.


6    Dosimetric Evaluation
The cost defined in equation 2 is an approximate measure of PTV underdosage. To assess its limitations
and fully assess the clinical potential of AF, we replanned six patients (3, 14, 18, 25, 26, 49) in the MR-
Linac treatment planning system, using the dose schedules proposed by the AF algorithm. Our goal was
to quantify the true dosimetric benefit, i.e., the additional dose delivered to the PTV compared to the
clinically delivered uniformly fractionated treatment. PTV underdosage is now measured by the actual
area above the DVH curve for the PTV in each fraction, where the area is integrated up to the prescribed
dose in the respective fraction. Summing these values across all fractions provided a total improvement
in PTV coverage in units of ccGy per patient. As an illustrative example, we first revisit patient 3.




(a) DVH of the first fraction. The uniformly fraction-        (b) DVH of the second fraction. The uniformly frac-
ated treatment has an area above the curve of 19.5            tionated treatment has an area above the curve of 5.4
ccGy between 0 and 6.6 Gy for the PTV, while the AF           ccGy between 0 and 6.6 Gy for the PTV, while the AF
DVH shows 6.2 ccGy between 0 and 6 Gy.                        DVH shows 19.2 ccGy between 0 and 9 Gy.

Figure 10: DVHs of the PTV (blue) and GTV (orange) for the uniformly fractionated (solid line) and
AF (dashed line) plans. Vertical lines indicate the prescribed AF doses (red) and uniform doses (black).
Black horizontal lines represent the fraction of the PTV volume that overlaps with the OAR.




                                                         17
                         (a)                                                   (b)

Figure 11: Re-planned dose distributions for patient 3. The blue contour represents the GTV, the red
contour the PTV, the green contour the dose-limiting OAR (in this case the bowel). a) Fraction 2. Due
to favorable overlap in this fraction, the algorithm suggested a high fraction dose of 9 Gy. Isodose lines
represent the dose as follows: Red = 9 Gy (100%), Orange = 7 Gy (78%), Yellow = 6 Gy (67%), Green
= 5.4 Gy (60%). Note there is no overlap between PTV and OAR in this slice due to interfraction
displacement of the bowel. b) Fraction 5. Due to unfavorable overlap in this fraction, the algorithm
suggested a low fraction dose of 6 Gy. Isodose lines represent the dose as follows: Orange = 7 Gy (117%),
Yellow = 6 Gy (100%), Green = 5.4 Gy (90%).

In Figure 10a, the DVHs for the first fraction are shown. The uniformly fractionated plan prescribed 6.6
Gy to the PTV, indicated by the vertical black line. The solid blue line shows the corresponding DVH.
Due to a large OAR-PTV overlap of 19.8 cc (approximately 30% of the PTV), only around 70% of the
PTV received at least 6 Gy, and about 60% reached the prescribed dose of 6.6 Gy. This results in a area
above the DVH curve of 19.5 ccGy.

In contrast, the AF plan for this fraction prescribed only 6 Gy (red vertical line), shown with a dashed blue
line. Although the spatial overlap remains the same, AF leads to a reduced area above the DVH of just
6.2 ccGy. The key here is that the evaluation only considers the dose up to the AF-prescribed dose of 6 Gy.

Figure 10b shows the DVHs for the second fraction, where the anatomical situation was more favorable:
the overlap was only 6.0 cc (about 10% of the PTV). Consequently, the PTV coverage improved in both
plans. The uniform plan resulted in a smaller area above the DVH of 5.4 ccGy. In the AF plan, a larger
dose of 9 Gy was prescribed, and the area above the DVH is 19.2 ccGy. This is still lower than the area
above the DVH for the uniformly fractionated plan in the first fraction. Figure 11b shows the dose distri-
bution obtained for fraction two, illustrating that higher dose of 9 Gy can be delivered to most of the PTV.

In Table 5, the area above the PTV curve for both the uniform and adaptive fractionation plan is listed
for all fraction. In the AF plan, only the second fraction receives a high dose of 9 Gy, while all other
fractions are prescribed the minimum dose of 6 Gy. Consequently, the AF plan incurs a dose coverage
disadvantage in the second fraction only. In all other fractions, AF results in less PTV underdosage com-
pared to uniform fractionation, most notably in the first (13.3 ccGy gain) and fourth (14.4 ccGy gain)
fraction, those with the largest PTV–OAR overlap. Overall, adaptive fractionation reduces the cumula-
tive PTV underdosage by 27.6 ccGy compared to uniform fractionation. Compared to the theoretically
predicted benefit of 22.4 ccGy, the true benefit even increased.




                                                     18
   Fraction     Overlap [cc]        PTV uniform                     PTV AF                Difference [ccGy]
                               Dose [Gy] Cost [ccGy]         Dose [Gy] Cost [ccGy]
       1            19.97         6.6        19.5                6          6.2                  -13.3
       2             6.02         6.6         5.4                9         19.3                  13.8
       3             9.45         6.6         9.5                6          3.2                   -6.2
       4            19.59         6.6        20.0                6          5.6                  -14.4
       5            12.62         6.6        13.3                6          5.8                   -7.5
           Total [ccGy]                 67.7                          40.1                       -27.6

Table 5: Area above the DVH curve for all fractions for patient 3, measured in ccGy. The AF plan shows
lower underdosage in four of the five fractions compared to the uniform plan. The difference column
quantifies the gain or loss in PTV dose coverage for AF relative to the uniform plan; negative values
indicate improved coverage.

In a second example, we examine patient 14, for whom the predicted benefit is 8.8 ccGy. Figure 12a
shows the DVHs for the second fraction. With an OAR-PTV overlap of 3.35 cc (approximately 7% of the
PTV), the uniform plan results in a modest PTV underdosage of 5.2 ccGy. The adaptive plan prescribes
a reduced dose of 6 Gy, leading to an even smaller underdosage of only 0.8 ccGy.

Figure 12b illustrates the third fraction, which has a minimal overlap of just 0.53 cc (about 1% of the
PTV). Here, the AF plan delivers a high dose of 10 Gy, with a resulting underdosage of 13.4 ccGy. In
comparison, the uniform plan at 7 Gy yields a cost of only 1.0 ccGy. Although the overlap is significantly
lower, the underdosage for a 10 Gy prescription is large in comparison to the underdosage in the uniformly
fractionated treatment in other fractions. The reason for this is illustrated in Figure 13a, showing the
corresponding dose distribution. Although the volume overlap is small, the bowel wraps around the ante-
rior part of the PTV. Because the dose gradient at the interface of bowel and PTV cannot be arbitrarily
steep, parts of the PTV at close distance from the bowel do not receive the prescribed dose of 10 Gy. In
Figure 12a this is seen as the falloff of the PTV-DVH at large doses, which contributed to the area above
the curve.

Table 6 summarizes the per-fraction PTV underdosage for both plans. While the AF plan achieves better
coverage in three of the five fractions, the 10 Gy prescription in fraction 3 yields a disproportionately
high cost. As a result, the total underdosage for AF (19.0 ccGy) is nearly identical to that of the uniform
plan (18.9 ccGy). The predicted benefit of 8.8 Gy could not be realized in this patient.




(a) DVH of the second fraction. The uniformly frac-        (b) DVH of the third fraction. The uniformly fraction-
tionated treatment has an area above the curve of          ated treatment has an area above the curve of 1.0 ccGy
5.2 ccGy between 0 and 7 Gy for the PTV, while the         between 0 and 7 Gy for the PTV, while the AF DVH
AF DVH shows 0.8 ccGy between 0 and 6 Gy.                  shows 13.4 ccGy between 0 and 10 Gy.

Figure 12: DVHs of the PTV (blue) and GTV (orange) for the uniformly fractionated (solid line) and AF
(dashed line) plans for patient 14. Vertical lines indicate the prescribed AF doses (red) and uniform doses
(black). Black horizontal lines represent the fraction of the PTV volume overlapping with the OAR.




                                                      19
   Fraction     Overlap [cc]           PTV uniform               PTV AF              Difference [ccGy]
                                  Dose [Gy] Cost [ccGy]   Dose [Gy] Cost [ccGy]
       1             3.35             7          4.5          6          1.0               -3.5
       2             3.35             7          5.2          6          0.8               -4.4
       3             0.53             7          1.0         10         13.4               12.4
       4             1.92             7          2.8          7          2.8                0
       5             3.66             7          5,4          6          1.0               -4.4
           Total [ccGy]                    18.9                    19.0                    0.1

Table 6: Area above the DVH curve (i.e., PTV underdosage) per fraction for patient 14, measured in
ccGy. The AF plan achieves improved coverage in most fractions, but the single high-dose fraction
introduces substantial underdosage. The final column shows the gain or loss in PTV coverage for AF
relative to the uniform plan; negative values indicate a benefit of AF.




                            (a)                                            (b)

Figure 13: Dose distributions for patient 14 for fraction 3. The blue contour represents the GTV, the
red contour the PTV, the light green contour the dose-limiting OAR (in this case the duodenum). a)
Re-planned fraction with the adapted dose of 10 Gy. Isodose lines represent the dose as follows: Orange
= 10 Gy (100%), Yellow = 8 Gy (80%), Green = 6 Gy (60%), Blue = 5 Gy (54%). Due to the finite dose
gradient between PTV and duodenum, a lack of PTV-coverage as well as issues with dose conformality
can be observed. b) Original plan for a 7 Gy prescription. Isodose lines represent the dose as follows:
Orange = 8 Gy (114), Yellow = 7 Gy (100%), Green = 6 Gy (86%), Blue = 5 Gy (77%). The shallower
dose gradient towards the duodenum allows for better PTV-coverage and conformality.

Table 7 summarizes the results for all six replanned patients. For most patients, the area above the DVH
is smaller for the AF plan compared to the clinically delivered uniformly fractionated plans, showing that
the benefit of AF can be realized. However, the benefit tends to be smaller compared to what is predicted
by the algorithm.

                                                                   Actual          Predicted
 Patient     Prescription [Gy]     Uniform [ccGy]   AF [ccGy]   Benefit [ccGy]   Benefit [ccGy]
   3                33                  67.7          40.1          -27.6            -22.4
   14               35                  18.9           19            0.1              -8.8
   18               35                   6.2           2.1           -4.1             -3.6
   25               35                   1.4           0.1           -1.3             -1.6
   26               35                  13.6           9.5           -4.1             -5.7
   49               40                  79.9          71.6           -8.3            -14.2

Table 7: Area above the DVH for all six replanned patients for the uniformly fractionated plan and the
AF plan.




                                                     20
7     Discussion
7.1    Summary of Main Findings
We consider challenging cases for MR-guided SBRT where the PTV prescription dose is higher than the
dose constraint for an OAR that overlaps with the PTV, requiring compromises in PTV coverage. We
introduced an approach to AF that exploits inter-fraction variation in the overlap volume between the
PTV and the dose-limiting OAR, aiming to minimize underdosage within the PTV. This is achieved by
delivering higher doses in fractions with favorable PTV/OAR geometries (i.e. small overlap volumes)
and lower doses on days with unfavorable geometries (i.e. large overlap volumes). The algorithm based
on MDPs is introduced to determine the daily dose delivered.

Across all 58 patients that were evaluated, AF provided a dosimetric benefit in a total of 55 cases that
showed reduced PTV underdosage compared to uniform fractionation. The mean benefit for all patients
was 2.9 ccGy, of which a small subset achieved more substantial improvements (up to 22.5 ccGy). The
magnitude of the observed benefits was strongly correlated with the degree of variation in PTV/OAR over-
lap, illustrating the relevance of this parameter as a dominant predictor of AF performance. Conversely,
patients with low or unfavorably ordered overlap variations exhibited limited benefit or even a dosimetric
disadvantage compared to uniform fractionation. Overall, these findings indicate that overlap-guided AF
can achieve moderate dosimetric improvements for most patients and larger benefits for selected cases
with pronounced anatomical variability.


7.2    Relation to Other Works
Previous works have introduced the concept of AF [10][11][12][13]. These methods have in common
that the fraction doses are adapted with the goal of exploiting daily anatomical variations. However,
the approach presented here differs regarding its goal and the situation in which is is applicable. Prior
works have quantified daily variations in terms of the sparing factor, defined as the ratio of OAR BED
and PTV BED, and the goal is to maximize the cumulative PTV BED [13]. However, for the sparing
factor to vary between fractions, the distance between PTV and OAR must change. In situations where
PTV and OAR overlap in all fractions, the sparing factor is approximately constant. Thus, these earlier
approaches to AF where not applicable in these situations. The variant of AF introduced here addresses
this shortcoming.

Other research groups have also explored ideas to adapt fraction doses with different objectives. In a
recent study by van Lieshout et al., the primary goal was optimizing treatment efficiency in the context
of online adaptive radiotherapy by adapting the number of delivered fractions[15]. In an in-silico cohort
of patients receiving SBRT for abdomino-pelvic lymph node oligometastases, patients showing advan-
tageous PTV/OAR geometry at treatment were re-planned using higher fraction doses with the goal of
shortening overall treatment duration while respecting OAR dose constraints. This endpoint of treatment
time reduction was reached in half of the included patients, shortening their treatment time from five to
one or two fractions. According to the authors, this concept will now be carried forward into a clinical trial.


7.3    Limitations and Future Work
The area above the DVH curve as a measure of PTV underdosage. In the AF algorithm, this is ap-
proximated by a square as illustrated in figure 5a. As described in section 6, the AF algorithm may
underestimate the true underdosage as measure by the area above DVH that is obtained after adaptive
replanning with the prescription doses obtained from the algorithm (Figure 13).
The reason for this is that the AF algorithm does not correctly account for the underdosage that occurs in
the non-overlapping part of the PTV due to the finite steepness of the dose gradient that can be achieved
at the interface to the OAR. The magnitude of this additional underdosage depends on the fraction dose
but also on the geometry of the target volume in relation to the dose-limiting OAR. Future work may
develop improved methods to estimate the area above the DVH, taking into account the falloff of the
DVH at large doses.




                                                      21
A practical difficulty regarding a future clinical implementation of AF in the form of a pilot phase-I-trial
concerns the observation that only relatively few patients have a substantial dosimetric benefit from AF
that is potentially clinically relevant. For most patients, the daily variation in the overlap volume turns
out to be too small. The ability to predict the amount of overlap variation and the benefit of AF for a
given patient a priori would therefore be helpful. For the patient cohort analyzed in this work, no clear
relationship between overlap variation and PTV volume or tumor site was apparent. Future work may
aim to identify surrogate parameters or indicators that can more reliably estimate AF benefit based on
pre-treatment imaging.



8    Conclusion
Overlap-variation-guided AF is an approach for optimizing PTV-coverage by adapting dose delivery to
daily variations in PTV/OAR geometry in patients receiving MR-guided SBRT for abdominal and pelvic
tumors with large interfractionatl motion that overlap with dose-limiting OARs. By assigning higher
doses to fractions with smaller overlap volumes and lower doses to those with larger ones, the method
improves target coverage without compromising on OAR sparing. While this approach to AF yields only
moderate dosimetric advantages in most patients, some cases with pronounced anatomical variability
show substantial benefit.


9    Appendix
The full implementation of the adaptive fractionation algorithm, together with a user interface, all analysis
scripts and anonymized data used in this study, is available on GitHub (https://github.com/YoelPH/
Overlap_adaptfx/tree/paper-version).




Figure 14: Mean PTV-volumes in the patient cohort. The colors represent the respective tumor locations.




                                                     22
Figure 15: Variation in overlap volumes for all patients. Each star represents the total calculated overlap
volume for one fraction, including the initial simulation. The red line represents the mean overlap volume
throughout each treatment course.


Table 8: AF plans, tumor sites, upper bounds, overlaps, overlapping OAR and benefits for all 58 patients.
PTV volume is given as average over all images. In cases of overlap with multiple dose-limiting OAR,
organs are ordered by overlap volume - largest to smallest. Abbreviations: Abd. unspec. = Abdominal
unspecified, B = Bowel, D = Duodenum, S = Stomach.

      Patient
    PTV Volume
    Prescription       Plan type                                                              Benefit
     Tumor Site          (OAR)         Planning       1       2        3       4       5      [ccGy]
         1               Overlap         2.02        2.41    3.04     1.96    5.3     3.27
        61.8               (B)
       33 Gy               AF                          6      6         9      6        6       3.71
      Pancreas        Upper Bound                      6       6       9       6        6       3.71
         2               Overlap          6.19       10.97   4.91     7.65    5.73    5.87
       124.5               (B)
       33 Gy               AF                          6      9         6      6       6        6.35
      Pancreas        Upper Bound                      6       9       6       6       6        6.35
         3               Overlap          9.08       19.79   6.02     9.45   19.59   12.62
        69.5              (B, S)
       33 Gy               AF                          6       9       6       6       6       22.42
      Pancreas        Upper Bound                      6       9       6       6       6       22.42
         4               Overlap          15.07      19.81   16.12   22.63   18.09   15.44
       106.9            (D, B, S)
       33 Gy               AF                          6       6       6       9       6        0.98
      Pancreas        Upper Bound                      6       6       6       6       9        8.93
         5               Overlap          17.15      18.68   18.35   28.56   21.37   22.98
       160.8            (D, B, S)
       33 Gy               AF                         6       6        6       6       9       -2.98
      Pancreas        Upper Bound                     6        9       6       6        6      10.91
         6               Overlap          0.25       0.53    0.99     2.66    2.41    0.27
        26.5               (B)
       35 Gy               AF                            6     6       6       7       10       3.37
      Pancreas        Upper Bound                        7     6       6       6       10       5.25


                                                    23
      7          Overlap     5.41    8.2     9.16   8.18   5.45   7.28
    41.3           (D)
   40 Gy           AF                 6       6      10     10      8    7.46
Abd. unspec.   Upper Bound            6        6     8      10     10    9.26
      8          Overlap     0.41    2.37    0.68   2.67   1.62   1.27
    80.0        (B, D, S)
   40 Gy           AF                 6       10     6      8      10    6.18
 Pancreas      Upper Bound            6       10     6      8      10    6.18
      9          Overlap     9.52   12.18    8.52   9.39   9.64   8.82
   109.6         (D, B)
   33 Gy           AF                 6       9      6       6     6     3.57
 Pancreas      Upper Bound            6       9      6       6      6    3.57
     10          Overlap     4.58    5.78    6.3    3.83   4.67   2.69
    68.4          (S, B)
   33 Gy           AF                 6       6      9       6     6     2.47
 Pancreas      Upper Bound            6        6     6       6      9    5.89
     11          Overlap     0.1     0.1     0.73   0.21   0.69   0.31
    29.9           (B)
   35 Gy           AF                  6      6      7      6      10    0.59
 Pancreas      Upper Bound            10      6      7      6       6    1.43
     12          Overlap      0      0.04     0     0.03    0     0.01
    1.5            (B)
   35 Gy           AF                 6       10     6      7      6     0.08
Lymph node     Upper Bound            6        9     6      9      6     0.08
     13          Overlap     4.79    6.28    8.71   6.39   5.97   8.3
   109.8           (S)
   35 Gy           AF                 6       6      7      10     6     5.38
Abd. unspec.   Upper Bound            7        6     6      10      6    5.49
     14          Overlap     2.88    3.35    3.35   0.53   1.92   3.66
    44.0           (D)
   35 Gy           AF                 6       6      10     7      6     8.77
Abd. unspec.   Upper Bound            6        6     10     7       6    8.77
     15          Overlap     1.01    1.57    1.57   2.88   1.64   1.52
    26.0           (B)
   35 Gy           AF                 6       6      6      10     7     1.1
Abd. unspec.   Upper Bound            6        7     6      6      10    1.53
     16          Overlap     1.38    3.64    6.73   3.42   5.99   9.17
   174.3          (B, S)
   40 Gy           AF                  6      6     10      8      10    -4.44
  Adrenal      Upper Bound            10      6     10      8       6    17.68
     17          Overlap     0.41    0.62    1.2    0       0     0.98
    27.3           (B)
   35 Gy           AF                 6       6      10     7      6      2.8
Abd. unspec.   Upper Bound            6        6     8      8       6     2.8
     18          Overlap     0.92    0.83    1.08   1.91    0     0.58
    22.5          (S, B)
   35 Gy           AF                    7    6      6      10     6     3.57
Abd. unspec.   Upper Bound               6     6     6      10     7     3.82
     19          Overlap      0          0   0.03    0     0.94    0
    27.7           (B)
   35 Gy           AF                 10      6      7      6      6     0.97
 Pancreas      Upper Bound            8        6     8      6       8    0.97
     20          Overlap     1.21    2.26    1.01    1     3.04   1.78
    11.7          (S, B)
   35 Gy           AF                 6       10     7       6     6     4.05
Lymph node     Upper Bound            6        7    10       6      6    4.08
     21          Overlap     4.33    3.93    7.84   4.3    7.23   6.89


                                    24
   35.7          (D)
  33 Gy          AF                 6       6      9      6      6     5.21
 Pancreas    Upper Bound            9        6     6      6       6    6.32
    22         Overlap      0      0.19    0.03    0     0.43   0.03
    3.8          (B)
  35 Gy          AF                 6       10      7     6      6     0.56
Lymph node   Upper Bound            6        7     10     6       6    0.65
    23         Overlap     3.68    4.56    3.57   5.24   4.51   2.59
   89.0         (S, B)
  35 Gy          AF                 6       10     6      6      7     3.6
Lymph node   Upper Bound            6        7     6      6      10    6.54
    24         Overlap     1.74    2.17    1.18   1.43   3.08   2.26
   10.5          (B)
  35 Gy          AF                 6       10     7      6      6     3.97
Lymph node   Upper Bound            6       10     7      6       6    3.97
    25         Overlap     0.41    0.44    0.75    0      0     0.38
   27.3          (B)
  35 Gy          AF                 6       6      10     7      6     1.57
Lymph node   Upper Bound            6        6     8      8       6    1.57
    26         Overlap     2.11    2.14    0.87   3.57   2.59   1.91
   12.9          (B)
  35 Gy          AF                 6       10     6      6      7     5.69
Lymph node   Upper Bound            6       10     6      6       7    5.69
    27         Overlap     5.35    5.56    3.85   8.76   5.95   5.59
   232.2      (D, S, B)
  33 Gy          AF                 6       9      6      6      6     6.28
 Pancreas    Upper Bound            6        9     6      6      6     6.28
    28         Overlap     2.75    2.53    2.77   2.48   3.6    4.5
   110.1      (D, S, B)
  33 Gy          AF                 6       6      9       6     6     2.09
 Pancreas    Upper Bound            6       6      9       6      6    2.09
    29         Overlap     1.92    4.48    2.2    1.87   2.54   2.47
   24.9          (B)
  35 Gy          AF                 6       7      10     6      6     3.88
 Pancreas    Upper Bound            6        7     10     6       6    3.88
    30         Overlap     5.41    7.95    6.37   5.62   7.43   8.73
   84.8         (D, S)
  33 Gy          AF                 6       6      9      6      6     4.8
 Pancreas    Upper Bound            6        6     9      6       6    4.8
    31         Overlap     2.16    2.98    3.76   2.52   3.92   3.55
   45.9          (D)
  33 Gy          AF                 6       6      9      6      6     2.48
 Pancreas    Upper Bound            6        6     9      6      6     2.48
    32         Overlap     1.56    0.48    0.22   0.19   0.33    0
   113.6        (B, S)
  40 Gy          AF                 10      10     8       6      6    -0.74
 Pancreas    Upper Bound            6        8    10       6     10    1.24
    33         Overlap     2.57    2.08    2.23   0.7    2.74   1.24
   19.6          (B)
  35 Gy          AF                 7       6      10     6      6     4.11
 Pancreas    Upper Bound            6        6     10     6       7    4.95
    34         Overlap     0.88    0.83    0.86   0.48   0.57   0.65
   97.0       (D, B, S)
  33 Gy          AF                 6       6      9      6      6     0.59
 Pancreas    Upper Bound            6        6     9      6       6    0.59
    35         Overlap     0.58    1.16    1.23   1.51   0.96   0.67
   17.6          (B)
  35 Gy          AF                    6    6      6     10      7     1.02


                                  25
 Pancreas      Upper Bound            6        6     6       7     10    1.89
     36          Overlap     0.3     0.39    0.36   0.81   0.12   0.15
    32.1          (D, S)
   33 Gy           AF                 6       6      6      9      6     0.74
 Pancreas      Upper Bound            6        6     6      9      6     0.74
     37          Overlap     0.91    0.19    0.66   0.56   0.7     0
    10.6           (B)
   35 Gy           AF                 10      6      6      6      7     1.35
Lymph node     Upper Bound            7        6     6      6      10    1.92
     38          Overlap     0.22    0.82    1.52   0.49   0.28   0.33
   166.3           (D)
   35 Gy           AF                    6    6      10     7      6     1.2
Lymph node     Upper Bound               6    6      6     10      7     1.99
     39          Overlap     0.17        0    0     0.25   0.6     0
    14.6         (D, B)
   35 Gy           AF                 10      7      6      6      6     0.85
Abd. unspec.   Upper Bound            8        8     6      6       8    0.85
     40          Overlap     1.2     0.26    1.07   1.62   1.2    0.66
    38.0           (S)
   35 Gy           AF                 10      6      6      6      7     3.11
Abd. unspec.   Upper Bound            10       6     6      6       7    3.11
     41          Overlap     0.19    0.04    0.25   0.02   0.06   0.15
    23.4           (D)
   35 Gy           AF                 10      6       7     6      6     0.34
 Pancreas      Upper Bound            7        6     10     6      6     0.4
     42          Overlap     0.73    0.98    0.69   0.83    1     1.4
    26.5           (B)
   35 Gy           AF                 6       10     6      6      7     0.74
  Adrenal      Upper Bound            6       10     7      6       6    1.31
     43          Overlap     1.18    1.48    0.56   2.45   0.18   1.07
   285.6          (B, S)
   35 Gy           AF                    6   10      6       7     6     3.32
  Adrenal      Upper Bound               6    7      6      10      6    4.46
     44          Overlap     0.14        0    0     0.02   0.13   0.02
    38.1           (D)
   40 Gy           AF                 10      10     8      6       6     0.3
Lymph node     Upper Bound            10      10     7       6      7     0.3
     45          Overlap     0.05    0.41    0.67   0.57   0.07   0.02
    53.1          (B, S)
   40 Gy           AF                 6       6      8      10     10    1.98
  Adrenal      Upper Bound            8        6     6      10     10    2.3
     46          Overlap     0.49    0.49    1.09   1.41   0.67   1.15
    5.0            (B)
   35 Gy           AF                 6       6      6      10     7     0.98
Lymph node     Upper Bound           10        6     6       7      6    2.18
     47          Overlap     1.91    2.3     1.52   2.01   2.19   2.97
    16.9           (B)
   35 Gy           AF                 6      10      6      6      7     1.94
Lymph node     Upper Bound            6      10      7      6      6     2.9
     48          Overlap      0      0.02     0     0.11    0      0
     37            (B)
   35 Gy           AF                 6       10     6      7      6     0.13
 Pancreas      Upper Bound            6        8     6      8       8    0.13
     49          Overlap     4.27    5.91    3.28   3.87   4.18   8.36
   124.3          (S, B)
   40 Gy           AF                    6    10    10      8       6    14.24
  Adrenal      Upper Bound               6    10    10       8      6    14.24
     50          Overlap     0.45        0   0.02    0     0.21   0.17


                                    26
       121.1             (D)
       35 Gy             AF                         10      7        6       6       6       0.38
      Adrenal        Upper Bound                    8        6       9       6       6       0.4
         51            Overlap           0.04        0     0.11      0       0       0
        64.0             (D)
       35 Gy             AF                         10      6       7       6       6        0.11
    Lymph node       Upper Bound                    7        6      7       7        7       0.11
         52            Overlap           0.53      0.01    0.03    0.24    0.54    0.53
        83.0             (S)
       35 Gy             AF                         10      7        6       6      6        1.28
      Adrenal        Upper Bound                    10       7       6       6       6       1.28
         53            Overlap           1.77      1.83    1.98     2.6    1.88    4.07
       222.9            (S, B)
       35 Gy             AF                          6      6       6       10      7        0.77
      Adrenal        Upper Bound                    10       6      6       7        6       3.16
         54            Overlap           1.48      1.52    0.97    1.65    0.41    0.36
        33.3             (S)
       40 Gy             AF                         6       10       6      10       8       3.58
      Adrenal        Upper Bound                    6        8       6      10      10       4.8
         55            Overlap            3        6.41    4.13    4.91    5.66    3.56
        66.0            (S, B)
       40 Gy             AF                         6       10       8      6       10       8.76
      Adrenal        Upper Bound                    6       10       8      6       10       8.76
         56            Overlap           0.59      0.01      0     0.32    0.17     0.6
        47.1             (S)
       40 Gy             AF                         10      10       6       8       6       1.82
      Adrenal        Upper Bound                    10      10       6       8       6       1.82
         57            Overlap           0.07      0.23      0       0       0     0.04
        37.1             (B)
       45 Gy             AF                         9       10      10      10       6       0.12
      Adrenal        Upper Bound                    6       10      10      10       9       0.69
         58            Overlap           0.24      0.65    0.21    0.12    0.06      0
        65.1             (D)
       40 Gy             AF                            6    10      10       8       6       0.64
      Adrenal        Upper Bound                       6     6      8       10      10       1.6


References
[1] Acharya, S. et al. Online magnetic resonance image guided adaptive radiation therapy: First clinical
    applications. International Journal of Radiation Oncology Biology Physics 94, 394–403 (2016). URL
    https://pubmed.ncbi.nlm.nih.gov/26678659/.
[2] Mutic, S. & Dempsey, J. F. The viewray system: Magnetic resonance–guided and controlled radio-
    therapy. Seminars in Radiation Oncology 24, 196–199 (2014). URL https://www.sciencedirect.
    com/science/article/pii/S1053429614000253?via%3Dihub.
[3] Raaymakers, B. W. et al. First patients treated with a 1.5 t mri-linac: clinical proof of concept
    of a high-precision, high-field mri guided radiotherapy treatment. Physics in Medicine & Biology
    62, L41 (2017). URL https://iopscience.iop.org/article/10.1088/1361-6560/aa9517https:
    //iopscience.iop.org/article/10.1088/1361-6560/aa9517/meta.
[4] Klüter, S. Technical design and concept of a 0.35 t mr-linac. Clinical and translational radiation
    oncology 18, 98–101 (2019). URL https://pubmed.ncbi.nlm.nih.gov/31341983/.

[5] Palacios, M. A. et al. Role of daily plan adaptation in mr-guided stereotactic ablative radiation
    therapy for adrenal metastases. International Journal of Radiation Oncology Biology Physics 102,
    426–433 (2018). URL https://pubmed.ncbi.nlm.nih.gov/29902559/.




                                                  27
 [6] Mayinger, M. et al. Benefit of replanning in mr-guided online adaptive radiation therapy in the
     treatment of liver metastasis. Radiation oncology (London, England) 16 (2021). URL https:
     //pubmed.ncbi.nlm.nih.gov/33947429/.

 [7] Pavic, M. et al. Mr-guided adaptive stereotactic body radiotherapy (sbrt) of primary tumor for pain
     control in metastatic pancreatic ductal adenocarcinoma (mpdac): an open randomized, multicentric,
     parallel group clinical trial (maspac). Radiation oncology (London, England) 17 (2022). URL
     https://pubmed.ncbi.nlm.nih.gov/35078490/.
 [8] Chuong, M. D. et al. Stereotactic mr-guided on-table adaptive radiation therapy (smart) for bor-
     derline resectable and locally advanced pancreatic cancer: A multi-center, open-label phase 2 study.
     Radiotherapy and Oncology 191, 110064 (2024). URL https://www.sciencedirect.com/science/
     article/pii/S0167814023093714?via%3Dihub.
 [9] Henke, L. et al. Phase i trial of stereotactic mr-guided online adaptive radiation therapy (smart) for
     the treatment of oligometastatic or unresectable primary malignancies of the abdomen. Radiotherapy
     and Oncology 126, 519–526 (2018). URL https://pubmed.ncbi.nlm.nih.gov/29277446/.
[10] Lu, W., Chen, M., Chen, Q., Ruchala, K. & Olivera, G.          Adaptive fractionation
     therapy:  I. basic concept and strategy.   Physics in Medicine & Biology 53, 5495
     (2008). URL https://iopscience.iop.org/article/10.1088/0031-9155/53/19/015https://
     iopscience.iop.org/article/10.1088/0031-9155/53/19/015/meta.

[11] Chen, M., Lu, W., Chen, Q., Ruchala, K. & Olivera, G.             Adaptive fractiona-
     tion therapy: Ii. biological effective dose. Physics in Medicine & Biology 53, 5513
     (2008). URL https://iopscience.iop.org/article/10.1088/0031-9155/53/19/016https://
     iopscience.iop.org/article/10.1088/0031-9155/53/19/016/meta.
[12] Ramakrishnan, J., Craft, D., Bortfeld, T. & Tsitsiklis, J. N.  A dynamic program-
     ming approach to adaptive fractionation.   Physics in Medicine & Biology 57, 1203
     (2012). URL https://iopscience.iop.org/article/10.1088/0031-9155/55/5/1203https://
     iopscience.iop.org/article/10.1088/0031-9155/55/5/1203/meta.
[13] Haas, Y. P., Ludwig, R., Bello, R. D., Tanadini-Lang, S. & Unkelbach, J. Adaptive fractionation at
     the mr-linac. Physics in Medicine & Biology 68, 035003 (2023). URL https://iopscience.iop.
     org/article/10.1088/1361-6560/acafd4.
[14] Sutton, R. S. & Barto, A. G. Reinforcement Learning: An Introduction (A Bradford Book, Cam-
     bridge, MA, USA, 2018).
[15] van Lieshout, E. et al. Reducing number of treatment fractions for patients with abdominal lymph
     node oligometastases: The need for online adaptive radiation therapy to provide personalized adap-
     tive fractionation. International Journal of Radiation Oncology Biology Physics 122, 721–728 (2025).
     URL https://pubmed.ncbi.nlm.nih.gov/40158733/.




                                                    28
