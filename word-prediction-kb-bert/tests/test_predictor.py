from itertools import islice
from typing import Tuple
import pytest
from sbx_word_prediction_kb_bert.predictor import TopKPredictor


TEXTS: dict[str, str] = {
    "short": "[MASK] åt glassen utanför kiosken .",
    "long-mask-middle": """
    Frisörbranschen påverkas väldigt negativt av den nuvarande situationen när kunderna inte kan komma till salongerna, vilket förstås gör att många frisörer är hotade av konkurs
    Totalt samlade lokalavdelningarna in 89 563 kronor för aktiviteter riktade till ungdomar under det första halvåret 2015, vilket var en ökning med 39 procent jämfört med samma period föregående år
    Utvärderingen visar att vi kan välja mellan att köpa in nya datorer som är mer kraftfulla och därför klarar de nya systemkraven eller att uppgradera våra datorer med nya grafikkort eller avstå från att installera den nya mjukvaran och
    Sammantaget bedömer vi att det kommer att vara [MASK] att genomföra projektet med befintlig personal och befintliga tekniska resurser och efter projektet kan vi starta förvaltningen av det system som projektet ska leda fram till
    Eftersom dessa rapporter är det huvudsakliga underlaget och empiriska materialet för utvärderingen som vi har genomfört har vi valt att i detalj beskriva hur utvärderingarna kommer till, vilka som genomför dem och hur deras slutrapporter ser ut
    Språkkonsulterna är en byrå av experter inom skriftlig och muntlig kommu­nikation – särskilt utbildade i att anpassa språket till den tänkta målgruppen
    Om jag bestämmer mig för att skriva en lång text så blir jag hungrig. Jag måste laga kaffe och kanske springa ner till bageriet, eller leta efter en karamell eller något annat sött. Och så när jag är klar med mina förberedelse ringer min före detta svärdotter. Hon har just blivit opererad. Det var blindtarmen och något mer som de plockade ut. Hennes mamma säger att hon gnäller och hennes nuvarande svärmorsa förstår inga europeiska språk, inte ett enda ord. Keine wörte bitte! Då sitter man där och ska vara snäll. Den nya svärdottern, hon som efterträdde den gamla, ringer aldrig. Kanske därför tar jag den gamla som en kompensation, eller en evig ursäkt för att inte skriva en lång text.
    Om jag nu bestämmer mig för att skriva en kort text, vad då? Då måste jag absolut börja med att kolla facebook, sedan mailboxen och sedan hur det går där borta i Sverige för de stackars bönderna som söker fruar men har så vansinnigt flotta och fina traktorer och hus som förr i världen skulle betraktas som herrgårdar, och så ringer telefonen. Det är en barndomsvän. Han brukar ringa varannan dag. Han har en ganska gedigen utbildning. Matematiker och gudarna vet vad, och sen fick han hjärnblödning och blev överkörd av en buss och ramlade omkull i Nationalteaterns trappuppgång, i den ordningen. Han orar sig för spionerna. Det finns tydligen spioner överallt. Det finns spioner utanför den amerikanska ambassaden och det finns spioner inni i den amerikanska ambassaden. Nu råkar denna stora vita byggnad, som hyser de amerikanska diplomaterna finnas här alldeles i närheten. De där stackarna som står givakt i alla väder kan åtminstone inte vara spionerna, men spionerna är kanske osynliga? I alla fall har jag fått reda på, genom en konstnär som spelar jultomte året om, att spionerna bakom den norska ambassaden är osynliga men om man står helt stilla kan man höra när de jojkar till varandra.
    Jag går och ser en föreställning och måste naturligtvis skriva lika långt varje gång. I går såg jag en ung man som åt upp ett päron, det var alltså för att vara orginell, för äpplet finns ju som en klassisk föreställning. Denna unga man som har långa ben, vansinnigt långa ben, ovanligt långa ben, lyckades tråka ihjäl femtiotvå personer som sprang ut ur en gammal fiskfabrik och kommer nog aldrig att se en s k föreställning där han finns med nån mer gång. Oj vad det är svårt att konsentrera sig om man har gäster. De ska ha kaffe och de ska underhållas, då kan man heller inte skriva en lång eller en kort text.
    """,
}


def test_short_text(kb_bert_predictor: TopKPredictor) -> None:
    text = "[MASK] åt glassen utanför kiosken ."

    actual = kb_bert_predictor.get_top_k_predictions(text)

    actual = remove_scores(actual)
    expected = "|Vi|Jag|De|Hon|Han|"

    assert actual == expected


def test_rounding(kb_bert_predictor: TopKPredictor) -> None:
    text = "namnet på det hus där historien börjar och slutar [MASK] men annars pratas det mest om huset . "
    kb_bert_predictor.num_decimals = 2
    actual = kb_bert_predictor.get_top_k_predictions(text)
    kb_bert_predictor.num_decimals = 3
    print(f"{actual=}")
    expected = [",:0.92", "-:0.06"]

    num_bars = actual.count("|")
    assert list(islice(actual.split("|"), 1, num_bars)) == expected


def remove_scores(actual):
    return "|".join(map(lambda x: x.split(":")[0], actual.split("|")))


def test_long_text(kb_bert_predictor: TopKPredictor) -> None:
    text = TEXTS["long-mask-middle"]

    assert len(text.split(" ")) > 512
    actual = kb_bert_predictor.get_top_k_predictions(text)

    actual = remove_scores(actual)
    expected = "|möjligt|lönsamt|lättare|fördelaktigt|lätt|"

    assert actual == expected


@pytest.mark.parametrize(
    "texts_key, expected", [("short", (0, 35)), ("long-mask-middle", (462, 864))]
)
def test_compute_context(
    texts_key: str, expected: Tuple[int, int], kb_bert_predictor: TopKPredictor
) -> None:
    actual_start, actual_end = kb_bert_predictor.compute_context(TEXTS[texts_key])

    expected_start, expected_end = expected

    assert actual_start == expected_start
    assert actual_end == expected_end
