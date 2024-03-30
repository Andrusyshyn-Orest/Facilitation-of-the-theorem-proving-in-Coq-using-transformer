(* This program is free software; you can redistribute it and/or      *)
(* modify it under the terms of the GNU Lesser General Public License *)
(* as published by the Free Software Foundation; either version 2.1   *)
(* of the License, or (at your option) any later version.             *)
(*                                                                    *)
(* This program is distributed in the hope that it will be useful,    *)
(* but WITHOUT ANY WARRANTY; without even the implied warranty of     *)
(* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      *)
(* GNU General Public License for more details.                       *)
(*                                                                    *)
(* You should have received a copy of the GNU Lesser General Public   *)
(* License along with this program; if not, write to the Free         *)
(* Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA *)
(* 02110-1301 USA                                                     *)


Require Import Reals.
Require Import trajectory_const.
Require Import trajectory_def.
Require Import constants.
Require Import ycngftys.
Require Import ycngstys.
Require Import tau.
Require Import ails.
Require Import trajectory.
Require Import measure2state.

Lemma d_distance :
 forall (intr : Trajectory) (evad : EvaderTrajectory),
 distance (measure2state (tr evad) 0) (measure2state intr 0) = d intr evad.
Proof with trivial.
unfold distance, d in |- *; unfold Die in |- *; simpl in |- *;
 unfold xi, yi in |- *...
Qed.

Lemma R_T :
 forall (intr : Trajectory) (evad : EvaderTrajectory) (T : TimeT),
 h (tr evad) = V ->
 Rsqr (RR (measure2state intr 0) (measure2state (tr evad) 0) T) =
 (Rsqr (l intr evad T * cos (beta intr evad T + thetat intr 0) - V * T) +
  Rsqr (l intr evad T * sin (beta intr evad T + thetat intr 0)))%R.
Proof with trivial.
intros intr evad T hyp_evad; rewrite Rsqr_minus...
unfold Rminus in |- *...
repeat rewrite Rplus_assoc...
set (z := 250%R)...
rewrite
 (Rplus_comm (Rsqr (l intr evad T * cos (beta intr evad T + thetat intr 0))))
 ...
repeat rewrite Rplus_assoc...
replace
 (Rsqr (l intr evad T * sin (beta intr evad T + thetat intr 0)) +
  Rsqr (l intr evad T * cos (beta intr evad T + thetat intr 0)))%R with
 (Rsqr (l intr evad T))...
rewrite <- (Rplus_comm (Rsqr (l intr evad T)))...
repeat rewrite <- Rplus_assoc...
replace
 (2 * (l intr evad T * cos (beta intr evad T + thetat intr 0)) * (V * T))%R
 with
 (2 * (l intr evad T * (V * T) * cos (beta intr evad T + thetat intr 0)))%R...
cut
 (l intr evad T =
  dist_euc (x (tr evad) T) (y (tr evad) T) (x intr 0%R) (y intr 0%R))...
intro...
rewrite H...
cut
 ((V * T)%R =
  dist_euc (x (tr evad) 0%R + T * z - T * z * cos (thetat intr 0))
    (y (tr evad) 0%R - T * z * sin (thetat intr 0)) 
    (x (tr evad) T) (y (tr evad) T))...
intro...
rewrite H0...
cut
 (RR (measure2state intr 0) (measure2state (tr evad) 0) T =
  dist_euc (x (tr evad) 0%R + T * z - T * z * cos (thetat intr 0))
    (y (tr evad) 0%R - T * z * sin (thetat intr 0)) 
    (x intr 0%R) (y intr 0%R))...
intro; rewrite H1...
replace
 (Rsqr
    (dist_euc (x (tr evad) 0 + T * z - T * z * cos (thetat intr 0))
       (y (tr evad) 0 - T * z * sin (thetat intr 0)) 
       (x (tr evad) T) (y (tr evad) T)) +
  Rsqr (dist_euc (x (tr evad) T) (y (tr evad) T) (x intr 0) (y intr 0)) +
  -
  (2 *
   (dist_euc (x (tr evad) T) (y (tr evad) T) (x intr 0) (y intr 0) *
    dist_euc (x (tr evad) 0 + T * z - T * z * cos (thetat intr 0))
      (y (tr evad) 0 - T * z * sin (thetat intr 0)) 
      (x (tr evad) T) (y (tr evad) T) *
    cos (beta intr evad T + thetat intr 0))))%R with
 (Rsqr
    (dist_euc (x (tr evad) 0 + T * z - T * z * cos (thetat intr 0))
       (y (tr evad) 0 - T * z * sin (thetat intr 0)) 
       (x (tr evad) T) (y (tr evad) T)) +
  Rsqr (dist_euc (x (tr evad) T) (y (tr evad) T) (x intr 0) (y intr 0)) -
  2 *
  (dist_euc (x (tr evad) T) (y (tr evad) T) (x intr 0) (y intr 0) *
   dist_euc (x (tr evad) 0 + T * z - T * z * cos (thetat intr 0))
     (y (tr evad) 0 - T * z * sin (thetat intr 0)) 
     (x (tr evad) T) (y (tr evad) T) * cos (beta intr evad T + thetat intr 0)))%R...
apply law_cosines...
unfold dist_euc in |- *...
replace
 (sqrt
    (Rsqr (x (tr evad) T - x intr 0%R) + Rsqr (y (tr evad) T - y intr 0%R)))
 with (l intr evad T)...
generalize (tr_cond1 evad (val T))...
intro; rewrite H2...
generalize (tr_cond2 evad (val T)); intro...
rewrite H3...
replace
 (x (tr evad) 0 + T * z - T * z * cos (thetat intr 0) -
  (x (tr evad) 0 + h (tr evad) * T))%R with (- T * z * cos (thetat intr 0))%R...
replace (y (tr evad) 0 - T * z * sin (thetat intr 0) - y (tr evad) 0)%R with
 (- T * z * sin (thetat intr 0))%R...
replace
 (Rsqr (- T * z * cos (thetat intr 0)) + Rsqr (- T * z * sin (thetat intr 0)))%R
 with (Rsqr (T * z))...
rewrite sqrt_Rsqr...
rewrite <- H2...
rewrite <- H3...
generalize xe_0; intro...
unfold xe, xi in H4...
rewrite (H4 intr evad T)...
generalize ye_0; intro...
unfold ye, yi in H5...
rewrite (H5 intr evad T)...
replace (x intr 0 - (l intr evad T * cos (beta intr evad T) + x intr 0))%R
 with (- (l intr evad T * cos (beta intr evad T)))%R...
unfold Rminus in |- *...
replace
 (y intr 0 + - (y intr 0 + - (l intr evad T * sin (beta intr evad T))))%R
 with (l intr evad T * sin (beta intr evad T))%R...
rewrite cos_plus...
unfold Rminus in |- *...
rewrite Rmult_plus_distr_l...
replace
 (- (l intr evad T * cos (beta intr evad T)) *
  (- T * z * cos (thetat intr 0)) +
  l intr evad T * sin (beta intr evad T) * (- T * z * sin (thetat intr 0)))%R
 with
 (l intr evad T * cos (beta intr evad T) * T * z * cos (thetat intr 0) +
  l intr evad T * - sin (beta intr evad T) * T * z * sin (thetat intr 0))%R...
replace (- (sin (beta intr evad T) * sin (thetat intr 0)))%R with
 (- sin (beta intr evad T) * sin (thetat intr 0))%R...
repeat rewrite <- Rmult_assoc...
repeat rewrite Rmult_assoc...
pattern (cos (beta intr evad T)) at 1 in |- *;
 rewrite (Rmult_comm (cos (beta intr evad T)))...
repeat rewrite <- Rmult_assoc...
replace
 (l intr evad T * T * z * cos (beta intr evad T) * cos (thetat intr 0))%R
 with
 (l intr evad T * T * z * cos (thetat intr 0) * cos (beta intr evad T))%R...
apply Rplus_eq_compat_l...
repeat rewrite Rmult_assoc...
apply Rmult_eq_compat_l...
repeat rewrite <- Rmult_assoc...
repeat rewrite <- (Rmult_comm (sin (thetat intr 0)))...
repeat rewrite Rmult_assoc...
apply Rmult_eq_compat_l...
rewrite (Rmult_comm (- sin (beta intr evad T)))...
repeat rewrite Rmult_assoc...
repeat rewrite Rmult_assoc...
repeat apply Rmult_eq_compat_l...
ring...
rewrite Ropp_mult_distr_l_reverse...
rewrite <- Ropp_mult_distr_l_reverse...
replace
 (- l intr evad T * cos (beta intr evad T) * (- T * z * cos (thetat intr 0)))%R
 with
 (l intr evad T * cos (beta intr evad T) * T * z * cos (thetat intr 0))%R...
apply Rplus_eq_compat_l...
repeat rewrite Rmult_assoc...
apply Rmult_eq_compat_l...
repeat rewrite <- Rmult_assoc...
repeat rewrite <- (Rmult_comm (sin (thetat intr 0)))...
apply Rmult_eq_compat_l...
repeat rewrite <- (Rmult_comm z)...
apply Rmult_eq_compat_l...
ring...
repeat rewrite <- Rmult_assoc...
repeat rewrite <- (Rmult_comm (cos (thetat intr 0)))...
apply Rmult_eq_compat_l...
repeat rewrite <- (Rmult_comm z)...
apply Rmult_eq_compat_l...
ring...
ring...
ring...
left; apply Rmult_lt_0_compat...
apply Rlt_le_trans with MinT...
apply MinT_is_pos...
apply (cond_1 T)...
unfold z in |- *; prove_sup...
replace (- T * z * cos (thetat intr 0))%R with
 (- T * (z * cos (thetat intr 0)))%R...
replace (- T * z * sin (thetat intr 0))%R with
 (- T * (z * sin (thetat intr 0)))%R...
repeat rewrite Rsqr_mult...
repeat rewrite <- Rsqr_neg...
rewrite cos2...
unfold Rminus in |- *; rewrite Rmult_plus_distr_l...
rewrite Rmult_1_r...
ring...
repeat rewrite Rmult_assoc...
repeat rewrite Rmult_assoc...
unfold Rminus in |- *...
rewrite (Rplus_comm (y (tr evad) 0%R))...
repeat rewrite Rplus_assoc...
rewrite Rplus_opp_r; rewrite Rplus_0_r...
repeat rewrite <- Ropp_mult_distr_l_reverse...
rewrite hyp_evad...
replace (v V) with z...
unfold Rminus in |- *...
ring...
unfold Rminus in |- *; unfold RR, dist_euc in |- *...
replace
 (sqrt
    (Rsqr (dx (measure2state intr 0) (measure2state (tr evad) 0) T) +
     Rsqr (dy (measure2state intr 0) (measure2state (tr evad) 0) T))) with
 (sqrt
    (Rsqr
       (x intr 0%R + T * z * cosd (toDeg (theta intr 0%R)) -
        (x (tr evad) 0%R + T * z)) +
     Rsqr
       (y intr 0%R + T * z * sind (toDeg (theta intr 0%R)) - y (tr evad) 0%R)))...
unfold cosd, sind, thetat in |- *...
rewrite rad_deg...
rewrite
 (Rsqr_neg
    (x intr 0%R + T * z * cos (theta intr 0%R) - (x (tr evad) 0%R + T * z)))
 ...
rewrite
 (Rsqr_neg (y intr 0%R + T * z * sin (theta intr 0%R) - y (tr evad) 0%R))
 ...
replace
 (- (x intr 0 + T * z * cos (theta intr 0) - (x (tr evad) 0 + T * z)))%R with
 (x (tr evad) 0 + T * z - T * z * cos (theta intr 0) - x intr 0)%R...
replace (- (y intr 0 + T * z * sin (theta intr 0) - y (tr evad) 0))%R with
 (y (tr evad) 0 - T * z * sin (theta intr 0) - y intr 0)%R...
unfold Rminus in |- *...
repeat rewrite Ropp_plus_distr...
rewrite Ropp_involutive...
rewrite <- (Rplus_comm (y (tr evad) 0%R))...
repeat rewrite Rplus_assoc...
apply Rplus_eq_compat_l...
rewrite (Rplus_comm (- y intr 0%R))...
unfold Rminus in |- *...
repeat rewrite Ropp_plus_distr...
repeat rewrite Ropp_involutive...
rewrite <- (Rplus_comm (x (tr evad) 0%R + T * z))...
repeat rewrite Rplus_assoc...
repeat apply Rplus_eq_compat_l...
rewrite (Rplus_comm (- x intr 0%R))...
cut (v V = z)...
intro...
rewrite H0...
unfold dist_euc in |- *...
generalize (tr_cond1 evad (val T)); intro...
generalize (tr_cond2 evad (val T)); intro...
rewrite H1...
rewrite H2...
rewrite hyp_evad...
rewrite H0...
replace
 (x (tr evad) 0 + T * z - T * z * cos (thetat intr 0) -
  (x (tr evad) 0 + z * T))%R with (- (T * z * cos (thetat intr 0)))%R...
replace (y (tr evad) 0 - T * z * sin (thetat intr 0) - y (tr evad) 0)%R with
 (- (T * z * sin (thetat intr 0)))%R...
repeat rewrite <- Rsqr_neg...
repeat rewrite Rsqr_mult...
rewrite cos2...
replace
 (Rsqr T * Rsqr z * (1 - Rsqr (sin (thetat intr 0))) +
  Rsqr T * Rsqr z * Rsqr (sin (thetat intr 0)))%R with 
 (Rsqr T * Rsqr z)%R...
rewrite <- Rsqr_mult...
rewrite sqrt_Rsqr...
apply Rmult_comm...
left; apply Rmult_lt_0_compat...
apply Rlt_le_trans with MinT...
apply MinT_is_pos...
apply (cond_1 T)...
unfold z in |- *; prove_sup...
unfold Rminus in |- *...
rewrite Rmult_plus_distr_l...
rewrite Rmult_1_r...
ring...
unfold Rminus in |- *...
rewrite (Rplus_comm (y (tr evad) 0%R))...
rewrite Rplus_assoc...
rewrite Rplus_opp_r...
symmetry  in |- *; apply Rplus_0_r...
unfold Rminus in |- *...
rewrite Ropp_plus_distr...
rewrite (Rplus_comm (- x (tr evad) 0%R))...
repeat rewrite Rplus_assoc...
rewrite (Rplus_comm (x (tr evad) 0%R))...
repeat rewrite Rplus_assoc...
rewrite Rplus_opp_l; rewrite Rplus_0_r...
ring...
unfold l, dist_euc in |- *...
unfold Die in |- *...
unfold xi, yi in |- *...
rewrite (Rsqr_neg (x intr 0%R - x (tr evad) T))...
rewrite (Rsqr_neg (y intr 0%R - y (tr evad) T))...
unfold Rminus in |- *...
repeat rewrite Ropp_plus_distr...
repeat rewrite Ropp_involutive...
rewrite <- (Rplus_comm (x (tr evad) T))...
rewrite <- (Rplus_comm (y (tr evad) T))...
repeat rewrite Rmult_assoc...
repeat apply Rmult_eq_compat_l...
rewrite <- (Rmult_comm (V * T))...
repeat rewrite Rmult_assoc...
repeat rewrite Rsqr_mult...
rewrite sin2...
unfold Rminus in |- *...
rewrite Rmult_plus_distr_l...
rewrite Rmult_1_r...
ring...
Qed.