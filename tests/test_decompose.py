"""Tests for claim decomposition."""

from claimverify.preprocessing.decompose import ClaimDecomposer


class TestClaimDecomposer:
    def setup_method(self):
        self.decomposer = ClaimDecomposer()

    def test_simple_claim_not_split(self):
        result = self.decomposer.decompose("Aspirin reduces the risk of colorectal cancer.")
        assert not result.is_compound
        assert result.n_parts == 1
        assert result.sub_claims[0] == "Aspirin reduces the risk of colorectal cancer."

    def test_comma_and_split(self):
        claim = ("APOE4 expression increases amyloid production, "
                 "and tau phosphorylation causes neuronal death.")
        result = self.decomposer.decompose(claim)
        assert result.is_compound
        assert result.n_parts == 2

    def test_verb_coordination(self):
        claim = "APOE4 expression increases amyloid production and promotes tau phosphorylation in neurons."
        result = self.decomposer.decompose(claim)
        assert result.is_compound
        assert result.n_parts == 2
        assert "APOE4 expression" in result.sub_claims[0]
        assert "APOE4 expression" in result.sub_claims[1]

    def test_short_fragments_not_split(self):
        claim = "X and Y are related."
        result = self.decomposer.decompose(claim)
        assert not result.is_compound

    def test_while_split(self):
        claim = ("Metformin reduces blood glucose levels in diabetic patients, "
                 "while insulin therapy maintains glycemic control through direct hormone replacement.")
        result = self.decomposer.decompose(claim)
        assert result.is_compound
        assert result.n_parts == 2

    def test_original_preserved(self):
        claim = "Some compound claim, and another part of the claim here."
        result = self.decomposer.decompose(claim)
        assert result.original == claim

    def test_batch_decompose(self):
        claims = [
            "Simple claim about aspirin and cancer risk.",
            "Complex claim increases production, and another claim causes degradation of cells.",
        ]
        results = self.decomposer.batch_decompose(claims)
        assert len(results) == 2
