# AdaptiveFractionation 2023 Paper

Physics in Medicine &
Biology



PAPER • OPEN ACCESS                                                                               You may also like
                                                                                                      - Optimal combined proton–photon therapy
Adaptive fractionation at the MR-linac                                                                  schemes based on the standard BED
                                                                                                        model
                                                                                                        S C M ten Eikelder, D den Hertog, T
To cite this article: Y Pérez Haas et al 2023 Phys. Med. Biol. 68 035003                                Bortfeld et al.

                                                                                                      - Toward semi-automatic biologically
                                                                                                        effective dose treatment plan optimisation
                                                                                                        for Gamma Knife radiosurgery
                                                                                                        Thomas Klinge, Hugues Talbot, Ian
View the article online for updates and enhancements.                                                   Paddick et al.

                                                                                                      - A feasibility study of spatiotemporally
                                                                                                        integrated radiotherapy using the LQ
                                                                                                        model
                                                                                                        M Kim and M H Phillips




                              This content was downloaded from IP address 31.10.159.189 on 01/03/2026 at 16:18
                             Phys. Med. Biol. 68 (2023) 035003                                                            https://doi.org/10.1088/1361-6560/acafd4




                             PAPER

                             Adaptive fractionation at the MR-linac
OPEN ACCESS
                             Y Pérez Haas∗ , R Ludwig             , R Dal Bello      , S Tanadini-Lang and J Unkelbach
RECEIVED
22 June 2022                 Department of Radiation Oncology, University Hospital of Zurich, Zurich, Switzerland
                             ∗
                               Author to whom any correspondence should be addressed.
REVISED
17 November 2022             E-mail: yoelhaas@gmail.com
ACCEPTED FOR PUBLICATION
3 January 2023
                             Keywords: adaptive fractionation, adaptive radiotherapy, Mr-linac, MR-guided radiotherapy, inter-fraction motion

PUBLISHED
19 January 2023
                            Abstract
Original content from this  Objective. Fractionated radiotherapy typically delivers the same dose in each fraction. Adaptive
work may be used under
the terms of the Creative   fractionation (AF) is an approach to exploit inter-fraction motion by increasing the dose on days when
Commons Attribution 4.0
licence.
                            the distance of tumor and dose-limiting organs at risk (OAR) is large and decreasing the dose on
Any further distribution of unfavorable days. We develop an AF algorithm and evaluate the concept for patients with abdominal
this work must maintain
attribution to the
                            tumors previously treated at the MR-linac in 5 fractions. Approach. Given daily adapted treatment
author(s) and the title of  plans, inter-fractional changes are quantiﬁed by sparing factors δt deﬁned as the OAR-to-tumor dose
the work, journal citation
and DOI.                    ratio. The key problem of AF is to decide on the dose to deliver in fraction t, given δt and the dose
                            delivered in previous fractions, but not knowing future δt s. Optimal doses that maximize the expected
                            biologically effective dose in the tumor (BED10) while staying below a maximum OAR BED3
                            constraint are computed using dynamic programming, assuming a normal distribution over δ with
                            mean and variance estimated from previously observed patient-speciﬁc δt s. The algorithm is evaluated
                            for 16 MR-linac patients in whom tumor dose was compromised due to proximity of bowel, stomach,
                            or duodenum. Main Results. In 14 out of the 16 patients, AF increased the tumor BED10 compared to
                            the reference treatment that delivers the same OAR dose in each fraction. However, in 11 of these 14
                            patients, the increase in BED10 was below 1 Gy. Two patients with large sparing factor variation had a
                            beneﬁt of more than 10 Gy BED10 increase. For one patient, AF led to a 5 Gy BED10 decrease due to an
                            unfavorable order of sparing factors. Signiﬁcance. On average, AF provided only a small increase in
                            tumor BED. However, AF may yield substantial beneﬁts for individual patients with large variations in
                            the geometry.

                             1. Introduction

                             Most radiation treatments are fractionated because normal tissue can tolerate higher doses if the radiation dose
                             is split into several fractions (Lajtha et al 1960, Fowler 2006). Motion of tumors and organs at risk (OAR) in
                             between fractions is generally assumed to degrade the quality of treatments. Traditionally, safety margins are
                             used to account for motion, which worsens the trade-off between tumor coverage and normal tissue sparing that
                             would be possible without motion (van Herk et al 2000).
                                  Nowadays, image guidance technology allows for measuring organ motion during the treatment, and the
                             treatment plan can be adapted to the changing anatomy (Mayinger et al 2021). Thereby, safety margins can be
                             reduced and, in the ideal case, such adaptive radiotherapy concepts restore the treatment quality that is possible
                             for a static patient without motion (Guckenberger et al 2011, Chen et al 2013, Brock 2019). This led to a
                             widespread implementation of stereotactic body radiation (SBRT)(Wulf et al 2006, Lo et al 2010, Andratschke
                             et al 2018).
                                  However, in principle inter-fraction tumor motion can even be exploited, that is, a better treatment quality
                             may be achieved in the presence of motion compared to a static patient geometry. Adaptive fractionation (AF)
                             (Chen et al 2008, Lu et al 2008, Ramakrishnan et al 2012) is one approach to exploit inter-fraction motion. In this
                             approach, a treatment plan is not only adapted to the daily patient geometry, but also the dose delivered to the
                             tumor in each fraction is modiﬁed: the dose is increased on favorable treatment days, i.e. when the distance


                             © 2023 The Author(s). Published on behalf of Institute of Physics and Engineering in Medicine by IOP Publishing Ltd
Phys. Med. Biol. 68 (2023) 035003                                                                                    Y Pérez Haas et al



between tumor and dose-limiting OAR is relatively large; and the dose is reduced for unfavorable geometries, i.e.
when the tumor and OAR are closer. Thereby, the ratio between total dose delivered to the OAR versus total dose
delivered to the tumor may be improved compared to uniform treatments that deliver the same dose in each
fraction.
    Although the idea of AF has been presented previously, clinical translation has been facing substantial
hurdles (Chen et al 2008, Lu et al 2008, Ramakrishnan et al 2012). AF relies on imaging to measure the changes of
the patient geometry from day to day (Wu et al 2002). This became possible to some degree with cone-beam CT.
However, the limited soft tissue contrast limits the potential clinical applications. Magnetic-resonance (MR)
guided radiotherapy (Acharya et al 2016, Henke et al 2018, Palacios et al 2018, Klüter 2019) allows the
visualization of anatomical variation with higher soft tissue contrast. This extends the range of potential
applications to abdominal lesions in proximity to bowel and stomach, which can exhibit very large inter-fraction
motion and may thus beneﬁt from AF. In this work, an approach for AF at an MR-linac is presented and
evaluated. The main contributions of this paper are:

1. To compute the optimal dose to deliver in each fraction, knowing today’s geometry and the dose delivered in
   previous fractions, but not knowing the patient geometry in future fractions, a dynamic programming
   algorithm has been developed - extending the work of (Ramakrishnan et al 2012).
2. The algorithm is tested on patients previously treated at the MR-linac with 5-fraction SBRT for abdominal
   lesions near bowel, stomach, or duodenum. Thus, the potential beneﬁt of AF is estimated for real patient
   data, extending previous work that only conceptually introduced AF based on synthetic data (Chen et al 2008,
   Lu et al 2008, Ramakrishnan et al 2012).


2. Methods and materials

2.1. Patients and treatment plans
We consider patients with abdominal tumors in proximity to either the bowel, stomach, or duodenum who
received 5-fraction SBRT treatments at the MR-linac system (MRIdian, Viewray). In all cases, tumor coverage
was compromised due to the dose received by the dose limiting OAR. All patients were planned and treated
according to institutional practice. In addition to the simulation MR and CT scans, daily MR scans were
performed for online adaptive radiotherapy. Tumors and OARs in a 2 cm ring around the tumor were
recontoured according to institutional guidelines and daily adaptive treatment plans were created. Thus, the
dose distributions were reoptimized in each fraction to adapt to inter-fractional changes, without altering the
prescription dose1. For the purpose of this project, dose-volume histograms (DVH) of 16 patients were exported
from the treatment planning system. DVHs were exported for GTV, PTV and the relevant OARs for 6 treatment
plans per patient, corresponding to the the 5 delivered plans and the initial plan based on the planning MR.

2.2. Sparing factors
For the purpose of AF, treatment plans and the geometric variations are described in terms of sparing factors δ:

                                                                  d tN
                                                           dt =        ,                                                           (1)
                                                                  dt

where dN t denotes the dose received by the dose-limiting OAR in fraction t and dt is the dose received by the
tumor. For the deﬁnition of dN t and dt, we follow the clinical practice of dose prescription and constraint
speciﬁcation: The dose to the OAR dN  t is deﬁned as the dose exceeded in 1cc of the OAR (D1cc), which represents
a commonly used dose parameter for the bowel, stomach or duodenum in SBRT treatments. The tumor dose dt
is deﬁned as the dose exceeded in 95% of the PTV volume (D95%), which is the dose parameter commonly used
for dose prescription and reporting (ICRU 1993, 1999). Each patient is thus described via a sequence of 6 sparing
factors corresponding to the planning MR and the 5 fractions. We further assume that inter-fraction motion is
random and is described by a Gaussian distribution over δ with a patient speciﬁc mean μ and standard deviation
σ

                                                       d t ~  ( m , s 2) .                                                        (2)



1
  GTV and PTV prescription doses and OAR constraints varied between patients. However, for all patients coverage was compromised due
to the OAR constraint.


                                                      2
Phys. Med. Biol. 68 (2023) 035003                                                                                          Y Pérez Haas et al



2.3. Fractionation
To model the fractionation effect, the biologically effective dose (BED) model is used (Jones et al 2001,
McMahon 2018). It is assumed that the classic BED model can be extended to varying doses per fraction such
that the cumulative BED at the end of treatment is given by the sum of the BED values delivered in individual
fractions. Thus, the cumulative BED delivered to the tumor is given by
                                                                 t            2
                                                                       dt ⎞
                                                   B T = å ⎛⎜d t +           ⎟,                                                          (3)
                                                         t=1 ⎝     ( a  b )T ⎠
where dτ denotes the dose delivered to the tumor in fraction τ. Consequently, the cumulative BED received by
the OAR is
                                                             t
                                                                      d t dt ⎞2   2
                                                 B N = å ⎛⎜dt d t +            ⎟,                                                        (4)
                                                       t=1 ⎝        ( a   b )N ⎠
where δτ is the sparing factor in fraction τ. In this work, the α/β ratios for the OARs and the tumors are the same
for all patients and set to (a b )N = 3 and (a b )T = 10 (van Leeuwen et al 2018). Correspondingly, the
biological effective doses in the OAR and the tumor will be denoted as BED3 and BED10 respectively. Note that
the calculation of cumulative BED3 in equation (4) assumes that the same 1 cc of the dose-limiting OAR receives
the highest dose, which may not be the case in reality. In this case, equation (4) can be considered a worst-case
measure for OAR dose, which overestimates the cumulative BED3 received by any part of the OAR. However,
due to the impracticality of deformable dose accumulation in the abdomen, the same approximation is done in
current clinical practice.

2.4. Adaptive fractionation
The goal of AF is to optimally decide on the doses dt delivered to the tumor in each fraction as to maximize the
expected cumulative BED10 delivered to the tumor, subject to a constraint on the cumulative OAR BED. For
5-fraction SBRT treatments in the abdomen, limiting the dose to bowel, stomach and duodenum is prioritized,
i.e. target coverage is compromised if necessary to fulﬁl the OAR constraint. This approach is clinical practice at
our own institution (Mayinger et al 2021) as well as other clinics (Tyagi et al 2021). For this work, we assume that
the BED delivered to the dose-limiting OAR may not exceed B Nmax= 90Gy BED3, which corresponds to 30 Gy
physical dose delivered in 5 uniform fractions (Pavic et al 2022). Thus, for patients in whom the tumor dose is
compromised due to the OAR constraint, as considered in this paper, the goal of AF is to increase tumor BED.2
      In each fraction t, the sparing factor δt can be determined. In addition, we know the BED delivered to the
tumor and the OAR in previous fractions. Based on this information, the decision must be made, which dose
should be delivered in today’s fraction. The difﬁculty comes from not knowing whether the remaining future
fractions will have favorable or unfavorable patient geometries. The sparing factors are random variables with an
estimated probability distribution but the exact future values are unknown.
      From a practical perspective, this approach to AF would be implemented by up-scaling or down-scaling the
reoptimized treatment plan for that fraction. That is, we assume that the adaptive radiotherapy process
consisting of MR imaging, recontouring, and plan reoptimization is not altered. The only additional step would
be a ﬁnal up-scaling or down-scaling of the ﬂuence without changing the shape of the dose distribution. This
corresponds to a renormalization of the plan, which is in current practice done within a narrow range that can be
extended to implement AF.                                       In which cases is this renormalisation done?




2.5. MDP model
To determine the optimal doses dt, we apply the framework of Markov decision processes (MDP) and formulate
AF as a stochastic optimal control problem. Here, we ﬁrst describe the MDP model for a known probability
distribution P(δt) and afterwards discuss how to estimate and update P(δt). Optimal control problems are
described by states, actions, state transitions and reward functions. In this application, these are given by:
    State: In each fraction, the state of a patient’s treatment is described by a two-dimensional vector s = (δ, BN)
that speciﬁes today’s sparing factor δ and the cumulative BED of the OAR that has been delivered so far in
previous fractions. Thus, the state of a treatment in fraction t for a patient with sparing factors {dt}tt= 1 treated
                  -1
with doses {d t}tt= 1 is
                                                          t-1
                                                                        d 2t d t2 ⎞ ⎞
                                              st = ⎛⎜dt , å ⎛⎜dt d t +            ⎟ ⎟.,                                                  (5)
                                                    ⎝ t=1 ⎝            (a b ) N ⎠ ⎠


2
 For applications of AF to other treatment sites where target coverage is prioritized over OAR sparing, the goal would instead be to minimize
expected cumulative OAR BED subject to a constraint on delivering the prescribed cumulative tumor BED.


                                                         3
Phys. Med. Biol. 68 (2023) 035003                                                                        Y Pérez Haas et al



    Action and policy: The actions correspond to the physical doses dt that are delivered to the tumor in a
fraction. Thus, a policy speciﬁes for each fraction t and possible state of the treatment, the dose that should be
delivered in this state. Tumor doses are constrained by a maximum dose per fraction d max and a minimum dose
per fraction d min . In the results below, we assume d min = 0 Gy and d max = 23 Gy . A single dose of 23 Gy allows
to deliver the BED10 equivalent of a typical prescription dose of 40 Gy in 5 fractions in a single fraction. This
range was chosen to explore the full potential of AF. The impact of tighter constraints on the dose per fraction,
which may be considered in a clinical implementation, is discussed in section 4.1.
    State transition: If, in fraction t, the treatment is in state st = (δt, B) and a dose dt is delivered to the tumor,
the state transitions to
                                                                        dt2 d t2 ⎞
                                       st + 1 = ⎛⎜dt + 1, B + dt d t +           ⎟,                                    (6)
                                                 ⎝                     (a b ) N ⎠
in fraction t + 1. The BED-component of the future state is calculated by adding the OAR BED delivered in
fraction t to the previously delivered BED B, which is assumed deterministic (i.e. we do not consider uncertainty
in dose delivery). The sparing factor in fraction t + 1 is random, making the state transition probabilistic. The
probability distribution for the state transition is simply given by the probability distribution over the sparing
factors, P(δ).
     Reward: In each fraction t, the immediate reward rt is given by the BED delivered to the tumor in that
fraction:
                                                                d t2
                                                 rt = d t +            .                                               (7)
                                                              (a b ) T
    To account for the cumulative BED constraint in the dose-limiting OAR, the BED must be below B Nmax . To
enforce this, a terminal reward of −∞ is assigned to all terminal states in which the cumulative OAR BED
delivered after the last fraction exceeds the constraint value.
    A characteristic of the speciﬁed MDP model is that the cumulative BED delivered to the tumor is not part of
the state s. It is only integrated in the reward rt. The reason for this is that the optimal policy does not depend on
the tumor state, i.e. the optimal dose to deliver does not depend on the previously accumulated tumor BED. The
intuitive explanation is that, in each state and fraction, we aim to maximize future tumor BED, independent of
the previously accumulated tumor BED. However, if the goal was to deliver a ﬁxed prescribed tumor BED by the
end of the treatment, the cumulative tumor BED would be part of the state (see also section 4.3).

2.6. Dynamic programming algorithm
A dynamic programming (DP) algorithm can be used to compute the optimal policy with the help of a value
function (Sutton and Barto 2018). The value function vt(δ, B) describes how desirable it is to be in state st(δ, B) in
fraction t and, therefore, it contains the information whether an action should be taken to reach that state. In this
application, the value for each state represents the expected cumulative BED that can be delivered to the tumor
in the remaining fractions, starting from that state and acting according to the optimal policy.
    The Bellman equation relates the value function in fraction t to the optimal policy and the value function in
the subsequent fraction, which for this application reads

                                           d2                                          d 2d 2 ⎞ ⎤
                   vt (d , B) = max ⎡d +          + å P (d ¢) vt + 1 ⎛d ¢ , B + dd +
                                                                           ⎜                   ⎟  ,                    (8)
                                    ⎢    (a b ) T                                    (a b ) N ⎠ ⎥
                                 d
                                    ⎣               d¢               ⎝                          ⎦
and

                                            d2                                          d 2d 2 ⎞ ⎤
                 pt (d , B) = argmax ⎡d +          + å P (d ¢) vt + 1 ⎛d ¢ , B + dd +
                                                                               ⎜                  .⎟                   (9)
                                     ⎢    (a b ) T                                    (a b ) N ⎠ ⎥
                                 d   ⎣               d¢               ⎝                          ⎦
     Value function and optimal policy can be calculated iteratively in one backward recursion starting from the
last fraction. To that end, the value function vF+1, corresponding to the terminal reward at the end of the
treatment after all F fractions are delivered, is initialized to
                                                              -¥ (B > B Nmax )
                                          v F + 1 (d , B) = ⎧                  .
                                                            ⎩0
                                                            ⎨    (B  B Nmax )

°The optimal policy is found by discretizing both actions and states. To reduce discretization artifacts in the
optimal policy, two tricks are applied. First, we use linear interpolation of the value function in the BED-
component of the state. Second, one can exploit that, in the optimal policy, the last fraction will simply deliver
the maximum residual BED3 to the OAR to end up at the cumulative BED3 constraint B Nmax . This can be used to
directly initialize value function and optimal policy in the last fraction F using continuous values of dF. Thereby,

                                                  4
Phys. Med. Biol. 68 (2023) 035003                                                                      Y Pérez Haas et al



artifacts from not reaching exactly the B Nmax in the last fraction due to discretization of the actions can be
avoided. The run time of the algorithm is in the order of seconds and therefore suited for on-line treatment
adaptation.

2.7. Probability updating
The DP algorithm relies on a description of the environment to compute an optimal policy, in this case the
probability distribution of the sparing factor P(δ), which we assume to be a Gaussian distribution truncated at 0,
with patient-speciﬁc parameters for mean and standard deviation. At the start of a treatment, only two sparing
factors are available for that patient, from the planning scan and the ﬁrst fraction. In each fraction, an additional
sparing factor is measured, which can be used to calculate updated estimates μt and σt for mean and standard
deviation, respectively.
    In each fraction t, we use the maximum likelihood estimator the mean of the sparing factor distribution
given by
                                                                        t
                                                                   1
                                                     mt =              å
                                                                 t + 1 t=0
                                                                           dt ,                                    (10)

where δ0 denotes the sparing factor from the planning MR. The estimator for the standard deviation, given the
patient-speciﬁc sparing factors up to fraction t, follows a chi-squared distribution, and the maximum likelihood
estimator is given by
                                                                    t
                                                               1
                                               stpat =             å (d t - m t ) 2 .
                                                             t + 1 t=0
                                                                                                                   (11)

    However, the standard deviation may be severely under- or overestimated if calculated from only two
samples at the very beginning of the treatment. Therefore, we assume a population based prior for the standard
deviation and compute the maximum a posterior estimator of σt via Bayesian inference. As the sparing factors
are assumed to follow a normal distribution with unknown variance, a gamma distribution is chosen as prior to
estimate the standard deviation σ
                                                                1                     -s
                                          f (s ; k , q ) =              s k - 1 exp ⎛    ⎞,                        (12)
                                                             G (k ) q k
                                                                                    ⎝ q ⎠
with shape parameter k and scale parameter θ. The maximum a posterior estimator for the standard deviation in
fraction t is then given by

                                                ⎡ s k-1       -s       ⎛ - (stpat )2 ⎞ ⎤
                                    st = argmax ⎢ t - 1 exp ⎛    ⎞ exp
                                                                       ⎜             ⎟ ⎥.                          (13)
                                                ⎢s          ⎝ q ⎠            s2
                                            s                               2t         ⎥
                                                ⎣                      ⎝             ⎠⎦
    Using equations (10) and (13), the probability distribution P(δ; μt, σt) is updated with every newly acquired
sparing factor and used in the Bellman equations (8) and (9) to recompute the optimal policy before each
fraction.

2.8. Quantiﬁcation of the beneﬁt
The treatment plan produced by AF is compared to three other treatments:

1. A reference treatment in which 6 Gy physical dose (18 Gy BED3) is delivered to the OAR in each fraction.
   Hence, the reference treatment delivers exactly the upper limit of 90 Gy BED3 to the OARs.
2. An upper bound for the beneﬁt of AF. To do so, we consider the hypothetical situation that all sparing factors
   δt are known before treatment. In that case, the optimal doses per fraction dt is calculated by solving the
   following optimization problem:
                                                                  F
                                                                                  d2
                                      maximize
                                           d
                                                                 å ⎛dt + (a/tb ) ⎞
                                                                      ⎜                  ⎟

                                                                 t =1 ⎝                 T⎠
                                                         F
                                                                          d2 d 2
                                      subject to:     å ⎛dt dt + (a/t bt) ⎞  B Nmax .
                                                             ⎜                      ⎟                              (14)
                                                      t =1 ⎝                       N⎠

    This treatment would optimally exploit the variation in δ and can thus be used to benchmark the beneﬁt of
    AF. However, it represents an unachievable upper bound for any realistic approach to AF where future
    sparing factors are unknown.

                                                         5
Phys. Med. Biol. 68 (2023) 035003                                                                                         Y Pérez Haas et al




   Figure 1. Scatter plot of all acquired sparing factors.




   Figure 2. Gamma distribution for the population-based prior for the standard deviation (solid line), estimated form the observed
   standard deviations in the 16 patients (dotted blue lines).




3. The clinically delivered treatment. The clinical treatment aims to deliver a ﬁxed prescription dose to the
   tumor in each fraction and may deliver less than 90 Gy BED3 to the OAR. Hence, the clinical treatment is
   included for qualitative comparison, while the reference treatment is used for quantitative evaluation of the
   beneﬁt of AF.

    To evaluate AF for a larger patient cohort, we generate additional patients by randomly sampling sparing
factors from distributions that resemble the 16 patients previously treated at the MR-linac.


3. Results

3.1. Observed sparing factors
Figure 1 shows the sparing factor distribution for each of the 16 patient and the respective dose-limiting OAR.
Substantial inter-patient variation is observed regarding the intra-patient variation of the sparing factors. Some
patients show substantial variation with a standard deviation of approximately 0.1 (e.g. patients 3, 5, 8, 13, 16),
whereas other patient show little variation of the sparing factor between fractions (e.g. patients 1, 2, 9). The

                                                             6
Phys. Med. Biol. 68 (2023) 035003                                                                                           Y Pérez Haas et al




   Figure 3. Optimal policy for fractions one to ﬁve assuming P(δ; μ1, σ1) with μ1 = 0.94 and σ1 = 0.058. (b) shows the value function of
   the ﬁrst fraction. The crosses mark the state of the treatment in each fraction, given the observed daily sparing factors and delivered
   doses for patient 7.




numerical values of the sparing factors as well as the DVHs from which they are computed are provided as csv-
ﬁles in the supplementary materials. The DVHs of patient 7, 8 and 13 are shown in appendix.
    The standard deviations of all patients were used to compute the hyperparameters of the gamma prior.
Figure 2 shows the resulting prior distribution, illustrating the inter-patient variation in the anatomical
variability.


3.2. Illustration of AF for an example patient
To illustrate AF based on the DP algorithm, patient 7 is discussed in detail. The sparing factors observed are:
δ0 = 0.88 for the planning MR and [0.99, 0.87, 0.98, 1.04, 1] for the 5 fractions. In the ﬁrst fraction, two sparing
factors are known, δ0 and δ1. These two sparing factors lead to the probability distribution P(δ; μ1, σ1) shown in
ﬁgure 3(a) (green dashed line), with a mean of μ1 = 0.94 and standard deviation σ1 = 0.058.

                                                         7
Phys. Med. Biol. 68 (2023) 035003                                                                                             Y Pérez Haas et al



Table 1. Dose delivered to the tumor for different treatments for patient 7. The results for AF in this table are computed with probability
distribution updating. Thus, the doses are not identical to the ones depicted in ﬁgure 3.

Fraction                        Sparing factor                Upper bound                 Adaptive fractionation                 Reference plan

Planning MR                          0.88                          —                                 —                                 —
First fraction                       0.99                          4.4                              4.1                                 6
Second fraction                      0.87                         12.6                              8.8                                6.9
Third fraction                       0.98                          4.7                              4.5                                6.1
Fourth fraction                      1.04                          3.3                              4.4                                5.8
Fifth fraction                       1.00                          4.1                              8.2                                 6
Total BED tumor                       —                           51.7                              50.2                               50



     Figure 3 shows the optimal policy obtained for this probability distribution. In the ﬁrst fraction, the OAR
BED is zero and the dose to deliver depends only on the sparing factor δ1 (ﬁgure 3(a), blue line). Since δ1 is
relatively high compared to the sparing factors to be expected given P(δ; μ1, σ1), a low dose of d1 = 4.1 Gy is
delivered to the tumor. In the second fraction, the dose to be delivered depends on the OAR BED delivered in the
ﬁrst fraction B and the sparing factor δ2 (ﬁgure 3(c)). The optimal policy is monotone in B and δ2, i.e. lower
sparing factors and lower previously accumulated BED lead to higher doses delivered in the current fraction.
Given the OAR BED3 of 9.6 delivered in the ﬁrst fraction and the sparing factor δ2 = 0.87, the optimal dose to
deliver in fraction 2 is d2 = 9.8 Gy, a rather high dose as δ2 is signiﬁcantly lower than the mean of the sparing
factor distribution and a low dose was delivered in the ﬁrst fraction. The structure of the optimal policy in the
third and fourth fraction is analogous to the second fraction (ﬁgures 3(d)/(e)). The dose delivered in the last
fraction (ﬁgure 3(f)) corresponds to the residual BED that can be delivered to the OAR to meet the constraint of
90 Gy BED3. In ﬁgure 3(b), the value function of the ﬁrst fraction is illustrated. For a sparing factor δ1 = 0.99, a
tumor BED10 of 53 Gy is expected based on the initial probability distribution.
     Figure 3 shows the optimal policy that was calculated based on the initial estimate of the probability
distribution, P(δ; μ1, σ1). By updating the probability distribution in each fraction as described in section 2.7, the
additional sparing factor observations can be incorporated into a better estimate of patient’s geometric
variability. In each fraction, the optimal policy is recalculated based on the updated probability distribution and
followed for the current fraction. E.g. in fraction two the sparing factor δ2 = 0.87 is observed. As a result, the
probability distribution is updated to P(δ; μ2, σ2) with μ2 = 0.91 and σ2 = 0.057. The reoptimized policy
proposes a dose of 8.8 Gy in fraction two. For patient 7, the change in the probability distribution between
fractions is small and consequently has only a small impact on the optimal policy, which is illustrated in ﬁgure 9
in appendix A.3.
     Table 1 summarizes the doses that would be delivered in each fraction and compares AF to the reference
treatment and the upper bound. In the reference treatment, the doses delivered to the tumor are 6/δt, resulting
in a cumulative tumor BED10 of 50.0 Gy. AF increases the tumor BED10 to 50.2 Gy, mainly by delivering a larger
dose in fraction 2, exploiting the lower sparing factor. At the beginning of the treatment, the achievable tumor
BED was estimated to be 53 Gy BED10. The resulting treatment plan delivered less BED to the tumor, as the
observed sparing factors were mostly higher than what was expected based on the ﬁrst two sparing factors.
     A comparison to the upper bound indicates that, for the sparing factor variations observed in this patient, the
possible improvement that AF may achieve is limited. By knowing all sparing factors in advance, one could have
further increased the dose delivered in fraction 2. However, even in this hypothetical case the tumor BED10
increases by only 1.7 Gy.

3.3. Evaluation of AF for all patients
The AF algorithm was applied to all 16 patients. Sparing factors and corresponding doses delivered in each
fraction are shown in ﬁgure 4 and reported in table 3 in the appendix. The quantitative comparison to the
reference treatment is provided in table 2.
    For 14 out of the 16 patients, AF yields an increase in tumor BED10 compared to the reference treatment.
However, for most patients, the difference is only around one Gray or less. For patients with small variations in
the sparing factor, such as patients 1, 2 and 9, there is little variation in the dose per fraction. Consequently, AF
performs similar to the reference treatment. In fact, the calculation of the upper bound shows that the beneﬁt of
AF is a priori limited to less than 0.3 Gy for these 3 patients. Patient 5 shows an intermediate variability in the
sparing factor. Here, AF realizes 1.6 Gy tumor BED10 increase, which corresponds to approximately half of the
upper bound for the improvement.
    Patients 3, 8 and 13 show large variations in the sparing factor and larger differences between AF and the
reference treatment. For patient 8, a sparing factor of 0.58 is observed in fraction 3, which is substantially lower

                                                          8
Phys. Med. Biol. 68 (2023) 035003                                                                                              Y Pérez Haas et al




    Figure 4. Sparing factors and corresponding tumor doses for each fraction for all patients. Different fractions are color coded; sparing
    factors are shown as crosses (x) according to the axis on the right; delivered doses are marked with a stars (*) according to the axis on
    the left.



                                Table 2. Comparison of treatment plans in Gy BED10. The difference column
                                is calculated as AF BED - reference plan BED, i.e. positive values indicate an
                                improvement using AF.

                                Patient          Reference         Upper           Adaptive
                                number           plan BED          bound         fractionation        Difference

                                1                   49.1            49.4              49.2               0.1
                                2                   55.6            55.7              55.6               0.1
                                3                   81.0           108.7             100.6               19.6
                                4                   49.7            50.6              50.2               0.5
                                5                   49.7            53.9              51.3               1.6
                                6                   43.4            43.8              43.5               0.1
                                7                    50             51.7              50.2               0.2
                                8                   71.3            93.1              85.5               14.2
                                9                    46             46.3              46.2               0.2
                                10                  67.3            68.8              67.5               0.2
                                11                  61.2             62               61.4               0.2
                                12                  54.1            55.3              54.9               0.8
                                13                  69.3           108.4              64.1               −5.2
                                14                   63             63.6              63.3               0.3
                                15                  69.1            70.9              69.7               0.6
                                16                  52.5             63               51.5               −1




compared to what was observed before. This is exploited by delivering a large dose of 20.2 Gy, which
corresponds to most of the residual BED3 that is allowed in the OAR. In fraction 4 and 5, the sparing factor is
again higher. Thus, delivering a large dose in fraction 3 was indeed a good decision, resulting in an improvement
of 14.2 Gy tumor BED10 compared to the reference treatment.3 Similarly, patient 3 had a very low sparing factor
in fraction 3, leading to a large tumor BED10 increase of 19.6 Gy.
     The clinically delivered treatment for patient 8 was based on a ﬁxed prescription of 7 Gy per fraction to the
PTV and an OAR constraint of 6 Gy per fraction. The DVHs are shown in ﬁgure 8(b) in DVH of clinically
delivered treatments. In the clinical treatment, the lower sparing factor in fraction 3 translates into a lower OAR
dose rather than an increased tumor dose. As a consequence, the accumulated BED3 in the OAR is only 76.4 Gy
and stays below the limit of 90 Gy. A cumulative BED10 of 59.2 Gy was delivered to the PTV.
3
  The impact of enforcing constraints on the minimum and maximum dose delivered to the tumor in each fraction is discussed in
section 4.1.


                                                             9
Phys. Med. Biol. 68 (2023) 035003                                                                                                     Y Pérez Haas et al




    Figure 5. Histogram of differences in BED 10 (adaptive fractionation minus reference treatment), for all 720 permutations of the
    sparing factors for (a) Patient 8, [0.77, 0.88, 0.8, 0.58, 0.86] and (b) Patient 13, [1.06, 0.92, 0.84, 0.82, 1.01, 0.53]. Highlighted are the
    120 permutations for which the lowest sparing factor was observed in the last fraction (red), the planning session (blue), and the
    second fraction (green).



3.4. Dependence on the order of sparing factors
Whereas patients 3 and 8 beneﬁt from the large variation in sparing factors through AF, we observe for patient 13
that the reference treatment performs better. This can be explained through the order of sparing factors, [1.06,
0.92, 0.84, 0.82, 1.01, 0.53], which is unfavorable for patient 13. The highest sparing factor is observed for the
planning MR and the lowest sparing factor is observed in the last fraction. The dose delivered in the last fraction
simply corresponds to the residual BED3 that can be delivered to the OAR. Since the algorithm cannot anticipate
an exceptionally low sparing factor in the last fraction, it does not lower the dose in previous fractions to exploit
the low sparing factor in the last fraction. In addition, observing the highest sparing factor in the planning MR,
leads to relatively large doses delivered in fractions 1 and 2. As a consequence, a residual dose of only 4.2 Gy can
be delivered to the OAR in the last fraction, resulting in worse performance than the reference treatment.
    The impact of the sparing factor order on the performance of AF is further analysed in ﬁgure 5 for patients 8
and 13. We consider permutations of the sparing factors, investigating hypothetical treatments in which the
same values were observed but in different order. When the lowest sparing factor is observed in the last fraction,
AF will in most cases yield lower cumulative tumor BED10 compared to the reference treatment. Instead, if the
lowest sparing factor is observed in the second or third fraction, AF improves on the reference treatment4.

3.5. Simulated patients
To quantify the beneﬁt of AF for a larger patient cohort, additional patients were generated by randomly drawing
sparing factors from a predeﬁned distribution. First, a patient speciﬁc mean μp and standard deviation σp was
selected for each generated patient p. The standard deviations σp were drawn from the gamma distribution
illustrated in ﬁgure 2 such that the variation of the sampled patients resembles the observed variation in our
cohort of 16 patients. Similarly, the patient speciﬁc means μp were drawn from a normal distribution that was
estimated based on the means of the 16 extracted patients. Subsequently, 6 sparing factors were drawn from a
normal distribution with the corresponding μp and σp. Figure 6 shows the histogram of tumor BED10 differences
between AF and the reference treatment for 5000 generated patients. The mean beneﬁt of AF for this patient
cohort is 0.93 Gy BED10. 81.2% of the sampled patients had a better treatment when AF was applied. The
histogram shows a long tail in the positive direction, corresponding to the relatively few patients that have a large
beneﬁt from AF. The less extended tail in negative direction shows that it is unlikely but possible that AF leads to
lower tumor BED10.
     To study the dependence of the beneﬁt of AF on the amount of variation of the sparing factors, patients from
11 different populations have been sampled. All populations have a constant mean sparing factor μ = 0.9; the
standard deviations is constant within each population but differs between populations. For each population,
the optimal policy was calculated assuming that the probability distribution is known a priori. Thus, the results
are slightly superior compared to the situation that the probability has to be estimated from the observed sparing
factors. Figure 7 shows the distribution of BED10 difference between AF and the reference treatment as a
4
  Note that permuting the sparing factors only affects the performance of AF whereas the cumulative tumor BED10 of the reference treatment
remains the same.


                                                             10
Phys. Med. Biol. 68 (2023) 035003                                                                                         Y Pérez Haas et al




   Figure 6. Histogram of the difference in tumor BED10 in Gy between adaptive fractionation and reference treatment for 5000 sampled
   patients.




   Figure 7. Box plot of BED10 differences between AF and the reference treatment. The larger the standard deviation of the sparing
   factor distribution, the larger the spread of the BED10 differences. The red line shows the median, blue lines the 25% and 75%
   percentiles, black lines a 1.5 fold extension of the interquartile range



function of the sparing factor standard deviation. Larger variation in the sparing factors is clearly correlated with
a larger mean beneﬁt from AF. Figure 7 also shows that the spread of BED10 differences is larger for increasing
standard deviations, i.e. more extreme treatments occur. Large beneﬁts from AF are more likely than treatments
that are substantially worse than the reference treatment, explaining the increase in the mean beneﬁt.


4. Modiﬁcations and extensions

The approach to AF described above can be modiﬁed and extended in different ways. In this section, selected
modiﬁcations are described.

                                                        11
Phys. Med. Biol. 68 (2023) 035003                                                                       Y Pérez Haas et al



4.1. Constraints on the dose per fraction
In the results above we did not consider a minimum tumor dose that must be delivered in each fraction. In
addition, the maximum dose was allowed to be high. In practice, one may want to limit the dose per fraction dt to
be within a range of clinically applied fractionation schemes. As an example, we consider a minimum dose of
3 Gy and a maximum dose of 15 Gy per fraction. For most of the 16 patients, this would lead to only minor
changes in the treatment since the optimal doses are within this range anyway (ﬁgure 4). For patients with large
sparing factor variations, modiﬁcations are seen. For patient 8, doses of [4.1, 8.2, 15, 3.4, 8.7] would be delivered,
resulting in a cumulative tumor BED10 of 78.9 Gy. Thus, constraints on the dose per fraction reduce the
potential beneﬁt of AF for such patients. For patient 13 these constraints on the dose per fraction result in an
increase of 0.5 Gy BED10 compared to the unconstrained AF plan due to a larger dose delivered in the last
fraction. Hence, the minimum dose does also reduce the potential of outcomes which are substantially inferior
compared to the reference plan.

4.2. Alternative deﬁnitions of the sparing factor
In this work, the sparing factor was deﬁned as the dose exceeded in 1 cc of the relevant OAR divided by the dose
exceeded in 95% of the PTV, which is motivated by two considerations: ﬁrst, clinical planning and reporting is
based on these parameters, and second, it can be assumed that the sparing factor could not have been improved
in the treatment plan optimization step. However, generally the deﬁnition of the sparing factor is ambiguous.
For example, the sparing factor can be deﬁned using the dose exceeded in 95% of the GTV rather than the PTV.
This would change the results of adaptive quantitatively but not qualitatively. A comparison is provided the
appendix A.5 in table 4.

4.3. Minimizing OAR BED for a given tumor prescription
In this paper, we considered patients in whom tumor BED was compromised due to the proximity to an OAR,
and thus the goal was to maximize cumulative tumor BED while respecting an OAR constraint. For patients with
lower sparing factors, a desired tumor prescription BTmin may be achieved. In this case, the objective may
become the minimization of OAR BED rather than a further increase of tumor BED. To account for that, the
MDP model can be extended to a three-dimension state
                                           t-1
                                                         d 2t d t2 ⎞ t - 1 ⎛      d t2 ⎞ ⎞
                               st = ⎛⎜dt , å ⎛⎜dt d t +            ⎟ , å ⎜d t +          ⎟ ⎟,                       (15)
                                     ⎝ t=1 ⎝            (a b ) N ⎠ t = 1 ⎝      (a b ) T ⎠ ⎠
where the third component is the tumor BED10 accumulated up to fraction t. The immediate cost rt in fraction t
is given by the BED delivered to the OAR in that fraction
                                                                  dt2 d t2 ⎞
                                               rt = - ⎛⎜dt d t +           ⎟,                                       (16)
                                                       ⎝         (a b ) N ⎠
and the terminal cost is

                                                 ⎧-¥                 (B N > B Nmax )
                         vF + 1 (d , B N , B T) = - k (B Tmin - B T) (B N  B Nmax and B T < B Tmin).
                                                 ⎨
                                                 ⎩0                  otherwise
The parameter κ weights the cost for underdosing the tumor. As long as κ is chosen large enough, the optimal
policy will prioritize delivering BTmin to the tumor over the objective of minimizing OAR BED.
    As an example, we assume a prescription BED10 of BTmin = 72 Gy , corresponding to 5 fractions of 8 Gy. For
the patients in whom the achievable BED10 is substantially lower, this extended DP algorithm yields almost
identical results as reported in ﬁgure 4. This is because, in the range of relevant values of δt and BN, the optimal
policy is almost independent of BT and approximately equal to the algorithm described in section 2.5. However,
for patient 8, the algorithm would deliver doses of [4.1, 8.3, 17.4, 0.9, 2] Gy, leading to a tumor BED10 of 72 Gy
and an OAR BED3 of 76.6 Gy.


5. Discussion

5.1. Summary of main ﬁndings
For SBRT treatments of abdominal lesions at the MR-linac, target coverage may be compromised when tumors
are located close to an OAR. In this work, we investigate whether AF may improve tumor BED. In current
practice, treatments are typically based on a ﬁxed prescription dose for each fraction. For tumors that are close to
an OAR in the planning MR scan, the prescription dose may be lowered compared to what would otherwise be
desired. If in some of the fractions a larger distance of tumor and OAR would allow for a larger tumor dose, the

                                                    12
Phys. Med. Biol. 68 (2023) 035003                                                                    Y Pérez Haas et al



prescription dose is not altered. Thus, the favorable geometry results in an OAR dose below the tolerance. One
step towards AF, called the reference treatment in this work, consists in increasing tumor and OAR dose up to
the per fraction dose constraint of the OAR. This may come with challenges for the clinical workﬂow and
treatment documentation but does not represent any scientiﬁc problem.
    In this work, we investigate if AF can improve the ratio of tumor versus OAR BED beyond the reference
treatment. To that end, the AF problem was formulated as a MDP and the optimal policy was determined via
dynamic programming. In contrast to prior work, we consider the problem that the probability distribution
over sparing factors is not known but has to be estimated for the individual patient. In addition, we evaluate the
algorithm using real data. To that end, we analysed 16 MR-linac patients previously treated for abdominal
lesions with 5-fraction SBRT. Main ﬁndings are:

1. The average beneﬁt of AF may be small. For the majority of patients that we analysed, the tumor BED10
   increase through AF was below 1 Gy. For these patients, the inter-fractional variation in geometry was too
   small, and calculation of an upper bound showed that no AF strategy may provide a signiﬁcant improvement.
2. Signiﬁcant improvements may be achieved for a subset of patients. For two patients with large variation in
   geometry, a substantial BED10 improvement in the order of 15 Gy could be achieved by delivering large doses
   when a favorable geometry occurs in the middle of the treatment.
3. Although improvements are more likely, there is a residual risk that AF yields inferior treatments. For a third
   patient with large geometry variation, AF resulted in 5 Gy lower tumor BED10. This problem occurs for an
   unfavorable order of sparing factors, e.g. if sparing factors in later fractions are substantially different from
   what was expected based on the initially estimated probability distribution. Presumably, no approach to AF
   may fully prevent this without reducing the beneﬁt of AF for other patients.

Simulations with randomly generated patients conﬁrmed the main ﬁndings from the 16 analysed patients.


5.2. Comparison to previous publications
Compared to the prior works by Lu et al (2008) and Chen et al (2008), we observe smaller beneﬁts of AF. Both
papers reported up to 30% decrease in OAR BED using AF. The average relative difference in tumor BED in this
work is in the order of a few percent. This large difference may originate from three key differences. The standard
deviations of the sparing factors observed for our MR-linac patients were notably lower than in the standard
deviations assumed in previous papers, where standard deviations between 0.1 and 0.6 were used to model the
variability of the sparing factor. Furthermore, the number of fractions is larger in both earlier papers, where the
prescribed dose to the tumor was delivered in 40 fractions, which gives more opportunity to the algorithm to
deliver lower doses on bad days and higher doses on good days. Additionally, in both earlier papers the
probability distribution of the sparing factors was assumed to be known a priori, which overestimates the quality
of the treatments compared to the real-world situation where the distribution has to be estimated for the
individual patient during treatment.


5.3. Coupling of treatment planning and fractionation decision
In this work, the AF problem was decoupled from the treatment planning problem. We used the clinically
delivered plans that were created based on ﬁxed prescription doses and OAR constraints, and assumed that AF is
performed by upscaling or downscaling the these treatment plans. In principle, one may expect improvements
by considering both problems jointly. This could well be done for the last fraction by setting the residual dose
that may be delivered to the OAR as the constraint. However, it is unclear how to combine treatment planning
and fractionation decision for earlier fractions.


6. Conclusion

Image guidance and daily replanning at the MR-linac enables a clinical implementation of AF as an approach to
exploit day-to-day variations in the distance of the tumor from the dose-limiting OAR. Based on our study
considering 5-fraction SBRT treatments of abdominal lesions in proximity to bowel, stomach or duodenum, we
conclude that for the majority of patients, the amount of interfraction motion may be too small to substantially
beneﬁt from AF. However, substantial tumor dose escalation may be achieved for a subset of patients with large
day-to-day changes of the geometry.

                                              13
Phys. Med. Biol. 68 (2023) 035003                                                                                          Y Pérez Haas et al



Appendix

A.1. Github repository
The algorithm and a graphical interface to apply AF are available in the GitHub repository adaptfx. The version
used for the results in this publication is made available as a persistent release (https://github.com/openAFT/
adaptfx/tree/perez_haas). This release also contains the DVHs of all 16 patients inside CSV ﬁles.


A.2. DVH of clinically delivered treatments




   Figure 8. DVH of GTV, PTV and the respective dose-limiting OAR for selected patients for the initial plan and the ﬁve fractions. For
   the OAR, only the 5cc receiving the highest dose is included in the DVH. Thus 1cc corresponds to 20%.



                                                        14
Phys. Med. Biol. 68 (2023) 035003                                                                                               Y Pérez Haas et al


A.3. Probability update




   Figure 9. (a) Update of the probability distribution of the sparing factors for patient 7 between fraction 1 and fraction 4. (b) Impact of
   the probability distribution update onto the optimal policy. Shown is the optimal dose to deliver in fraction 4 as a function of the
   sparing factor for a previously delivered OAR BED3 given by 47.6, showing differences of up to approximately 1 Gy.



A.4. Sparing factors and doses per fraction


      Table 3. Delivered doses in adaptive fractionation and the treatment corresponding to the upper bound.

      Patient        Plan type            Planning        Fraction 1        Fraction 2       Fraction 3        Fraction 4       Fraction 5

      1              sparing factor         0.99               0.95            0.98             0.96              1.02             1.01
                     adaptive                                   6.8             5.7              6.5                5               6.5
                     upper bound                                7.7              6               7.1               4.6              4.9

      2              sparing factor          0.9               0.91            0.92             0.88              0.9              0.89
                     adaptive                                   6.3             6.2              7.8              6.5               6.5
                     upper bound                                 6              5.6              7.9              6.6               7.2

      3              sparing factor         0.73               0.67            0.78             0.53              0.89             0.73
                     adaptive                                  10.8             3.9             21.8               0.3              2.4
                     upper bound                                1.6             0.4             27.9                0               0.8

      4              sparing factor         0.95                1              0.92             0.92              1.04             1.01
                     adaptive                                  5.1              7.9               7                 4               6.5
                     upper bound                               4.7              8.7              8.7               3.7              4.4

      5              sparing factor         0.94               0.83            0.97             1.13              1.05             0.96
                     adaptive                                   9.3             3.9              2.4               5.1               9
                     upper bound                               14.9              4               1.7               2.5              4.2

      6              sparing factor         1.04               1.03            1.1              1.04              1.14             1.08
                     adaptive                                   5.9            4.5               6.3               4.4              6.6
                     upper bound                                7.3            4.6               6.7               3.8              5.2

      7              sparing factor         0.88               0.99            0.87             0.98              1.04               1
                     adaptive                                   4.4             8.8              4.5               4.4              8.2
                     upper bound                                4.4            12.6              4.7               3.3              4.1

      8              sparing factor         0.77               0.88            0.8              0.58              0.86             0.77
                     adaptive                                    4             8.2              20.2                1               2.3
                     upper bound                               0.49            1.1              25.3               0.6              1.4

      9              sparing factor         1.04               1.02            0.99             1.02              1.08             1.05
                     adaptive                                   6.1             6.7              5.5               4.6              6.1
                     upper bound                                 6              7.6               6                4.2               5




                                                          15
Phys. Med. Biol. 68 (2023) 035003                                                                             Y Pérez Haas et al



      Table 3. (Continued.)
      Patient       Plan type           Planning       Fraction 1      Fraction 2   Fraction 3   Fraction 4   Fraction 5


      10            sparing factor        0.86              0.78          0.79        0.81         0.83          0.73
                    adaptive                                 9.4           7.7         6.5          6.2           8.2
                    upper bound                              6.8           6.1          5           4.4           14

      11            sparing factor        1.01              0.83           0.8        0.82         0.85          0.91
                    adaptive                                10.6           8.7         6            4.9           4.4
                    upper bound                              7.2          10.2         8            5.9            4

      12            sparing factor        0.88              0.94          0.88        0.97         0.96          0.85
                    adaptive                                 5.1           7.7         4.7          6.1           8.9
                    upper bound                              4.8           7.9         3.9          4.2          10.9

      13            sparing factor        1.06              0.92          0.84        0.82         1.01          0.53
                    adaptive                                 8.5           9.6         7.1          2.6           7.9
                    upper bound                               0            0.1         0.2           0           28.3

      14            sparing factor        0.84              0.85          0.83        0.78         0.82          0.84
                    adaptive                                 6.7           7.8         9.7          6.1           5.9
                    upper bound                              5.2           6.3        11.4           7            5.7

      15            sparing factor        0.78              0.72          0.74        0.83         0.79          0.79
                    adaptive                                 9.9           7.7         4.1          7.3           9.3
                    upper bound                             13.3           9.6         3.7          5.3           5.3

      16            sparing factor        0.98              1.05          1.06        1.06          0.9          0.74
                    adaptive                                 4.7            5          5.5          9.8           5.8
                    upper bound                              1.3           1.3         1.3           3           18.8




   Figure 10. PTV and the corresponding GTV sparing factors of all patients




A.5. Results for GTV based sparing factors
The same analysis as described in sections 2 and 3 can be conducted with GTV based sparing factors, i.e. all
sparing factors are deﬁned via the D95 of the GTV rather than the PTV, and the goal is to maximize BED10 in the
GTV. Figure 10 shows the relation of GTV and PTV sparing factors. As expected, GTV and PTV sparing factors
are correlated. Consequently, AF yields similar treatments for both deﬁnitions of the sparing factor. Table 4
(column 1) reports BED10 differences in the GTV compared to the reference treatment for AF optimized for
GTV-based sparing factors. The same treatment can then be evaluated for BED10 in the PTV (column 2). For

                                                       16
Phys. Med. Biol. 68 (2023) 035003                                                                                          Y Pérez Haas et al



                 Table 4. Plan differences based on GTV optimization (columns one and two) and PTV optimization
                 (columns three and four). The optimal doses in the ﬁrst two columns have been computed based on the
                 GTV sparing factors and based on the PTV sparing factors in columns three and four. The respective
                 differences to the reference plans are given in BED10.

                                                      GTV optimized                               PTV optimized
                 Optimization type
                 Patient number           GTV difference        PTV difference        PTV difference       GTV difference

                 1                              0.4                   −0.2                  0.1                    0.3
                 2                               0                      0                    0                      0
                 3                             21.8                   11.7                 19.6                    36
                 4                              4.6                    0.2                  0.5                    2.3
                 5                              1.5                    1.8                  1.6                    0.6
                 6                              0.6                   −0.2                  0.1                   −0.8
                 7                             −0.8                   −0.4                  0.2                   −0.2
                 8                             17.7                   13.6                 14.2                   18.6
                 9                             −2.6                   −3.1                  0.2                    0.7
                 10                              1                     0.3                  0.2                    0.6
                 11                            −0.4                     0                   0.1                   −0.2
                 12                             1.5                     1                   0.8                    0.9
                 13                            13.8                    8.2                 −5.2                   −6.9
                 14                            −2.1                   −0.7                  0.3                    0.4
                 15                             0.7                   −0.2                  0.6                    1.3
                 16                             2.8                   −0.3                 −1                     −0.2

                 Mean                           3.8                    2                     2                    3.4




comparison, table 4 also reports BED10 differences in the PTV (column 3) and the GTV (column 4) for AF
optimized for PTV-based sparing factors.
    Comparing results based on GTV and PTV based sparing factors, the most notable difference is seen for
patient 13, where AF based on PTV sparing factors leads to lower BED10 than the reference treatment. For GTV
based sparing factors, we instead see an improvement through AF, even though the lowest sparing factor is still
observed in the last fraction. However, there is overall less variation in the GTV sparing factors up to fraction 4,
the initial sparing factor better represents the mean, and the sparing factor in fraction 4 was higher than the other
sparing factors, leading to a small dose to be delivered in fraction 4. In combination, this results in a larger
residual dose available in fraction 5.


ORCID iDs

Y Pérez Haas https://orcid.org/0000-0002-4282-0643
R Ludwig https://orcid.org/0000-0001-9434-328X
R Dal Bello https://orcid.org/0000-0002-8755-377X

References
Acharya S et al 2016 Online magnetic resonance image guided adaptive radiation therapy: ﬁrst clinical applications Int. J. Radiat.
      Oncol.*Biol.*Phys. 94 394–403
Andratschke N et al 2018 The SBRT database initiative of the German Society for Radiation Oncology (DEGRO): patterns of care and
      outcome analysis of stereotactic body radiotherapy (SBRT) for liver oligometastases in 474 patients with 623 metastases BMC Cancer
      18 283
Brock K K 2019 Adaptive radiotherapy: moving into the future Semin. Radiat. Oncol. 29 181–4
Chen M, Lu W, Chen Q, Ruchala K and Olivera G 2008 Adaptive fractionation therapy: II. Biological effective dose Phys. Med. Biol. 53
      5513–25
Chen W, Gemmel A and Rietzel E 2013 A patient-speciﬁc planning target volume used in ’plan of the day’ adaptation for interfractional
      motion mitigation J. Radiat. Res. 54 i82–90
Fowler J F 2006 Development of radiobiology for oncologya personal view Phys. Med. Biol. 51
Guckenberger M, Wilbert J, Richter A, Baier K and Flentje M 2011 Potential of adaptive radiotherapy to escalate the radiation dose in
      combined radiochemotherapy for locally advanced nonsmall cell lung cancer Int. J. Radiat. Oncol.*Biol.*Phys. 79
Henke L et al 2018 Phase I trial of stereotactic MR-guided online adaptive radiation therapy (SMART) for the treatment of oligometastatic or
      unresectable primary malignancies of the abdomen Radiother. Oncol. 126 519–26
ICRU 1993 Prescribing, recording, and reporting photon beamtherapy ICRU Report 50 3–26
ICRU 1999 Prescribing, recording, and reporting photon beamtherapy (supplement to ICRU report 50) ICRU Report 62 3–20
Jones B, Dale R G, Deehan C, Hopkins K I and Morgan D A L 2001 The Role of Biologically Effective Dose (BED) in clinical oncology Clin.
      Oncol. 13 71–81



                                                        17
Phys. Med. Biol. 68 (2023) 035003                                                                                           Y Pérez Haas et al



Klüter S 2019 Technical design and concept of a 0.35 T MR-linac Clin. Trans. Radiat. Oncol. 18 98–101
Lajtha L G, Oliver R and Ellis F 1960 Rationalisation of fractionation in radiotherapy Br. J. Radiol. 33 634–5
Lo S S et al 2010 Stereotactic body radiation therapy: a novel treatment modality Nat. Rev. Clin. Oncol. 7 44–54
Lu W, Chen M, Chen Q, Ruchala K and Olivera G 2008 Adaptive fractionation therapy: I. Basic concept and strategy Phys. Med. Biol. 53
       5495–511
Mayinger M et al 2021 Beneﬁt of replanning in MR-guided online adaptive radiation therapy in the treatment of liver metastasis Radiat.
       Oncol. 16 84
McMahon S J 2018 The linear quadratic model: usage, interpretation and challenges Phys. Med. Biol. 64 01TR01
Palacios M A et al 2018 Role of daily plan adaptation in MR-guided stereotactic ablative radiation therapy for adrenal metastases Int. J.
       Radiat. Oncol.*Biol.*Phys. 102 426–33
Pavic M et al 2022 Mr-guided adaptive stereotactic body radiotherapy (sbrt) of primary tumor for pain control in metastatic pancreatic
       ductal adenocarcinoma (mpdac): an open randomized, multicentric, parallel group clinical trial (maspac) Radiat. Oncol. 17
Ramakrishnan J, Craft D, Bortfeld T and Tsitsiklis J N 2012 A dynamic programming approach to adaptive fractionation Phys. Med. Biol. 57
       1203–16
Sutton R S and Barto A G 2018 Reinforcement Learning: An Introduction (Cambridge, MA: MIT Press) 2nd edn
Tyagi N et al 2021 Feasibility of ablative stereotactic body radiation therapy of pancreas cancer patients on a 1.5 tesla magnetic resonance-
       linac system using abdominal compression Phys. Imaging Radiat. Oncol. 19 53–9
van Herk M, Remeijer P, Rasch C and Lebesque J V 2000 The probability of correct target dosage: dose-population histograms for deriving
       treatment margins in radiotherapy Int. J. Radiat. Oncol.*Biol.*Phys. 47 1121–35
van Leeuwen C M et al 2018 The alfa and beta of tumours: a review of parameters of the linear-quadratic model, derived from clinical
       radiotherapy studies Radiat. Oncol. 13 96
Wu C, Jeraj R, Olivera G H and Mackie T R 2002 Re-optimization in adaptive radiotherapy Phys. Med. Biol. 47 3181–95
Wulf J et al 2006 Stereotactic radiotherapy of primary liver cancer and hepatic metastases Acta Oncol. 45 838–47




                                                         18
