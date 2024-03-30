Require Export E3_Congruence.

Section CONTINUITY.

(* D1 (Archimedes) : Given line segments AB and CD, there is a natural number n such that n copies of AB added together will be greater than CD. *)

Lemma D1 : forall A B C D : Point, 
	A <> B ->
	C <> D ->
	exists n : nat,
		exists E : Point,
			HalfLine A B E /\
			Distance A E = DistTimesn n A B /\
			LSlt (Distance C D) (Distance A E).
Proof.
	intros.
	destruct (ExistsHalfLineEquidistant A B C D H H0) as (F, (H1, H2)).
	destruct (Archimedian A B F H) as (n, H3).
	destruct (Graduation A B H n) as (E, (H4, H5)).
	exists n; exists E; intuition.
	 apply HalfLineSym.
	  apply (LSltDistinct A F A E); rewrite H5; trivial.
	  trivial.
	 rewrite <- H2; rewrite H5; trivial.
Qed.

End CONTINUITY.
