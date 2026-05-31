const emojis = ['🍎', '🍌', '🍓', '🍇', '🍉', '🍒', '🥝', '🍍'];
let cards = [...emojis, ...emojis];
cards.sort(() => 0.5 - Math.random());

const gameBoard = document.getElementById('gameBoard');
let firstCard = null;
let secondCard = null;
let lock = false;
let matches = 0;

cards.forEach(emoji => {
  const card = document.createElement('div');
  card.classList.add('card');
  card.dataset.emoji = emoji;
  card.textContent = '?';
  card.addEventListener('click', () => flipCard(card));
  gameBoard.appendChild(card);
});

function flipCard(card) {
  if (lock || card.classList.contains('flipped')) return;

  card.textContent = card.dataset.emoji;
  card.classList.add('flipped');

  if (!firstCard) {
    firstCard = card;
  } else {
    secondCard = card;
    lock = true;

    if (firstCard.dataset.emoji === secondCard.dataset.emoji) {
      matches++;
      firstCard = null;
      secondCard = null;
      lock = false;

      if (matches === emojis.length) {
        document.getElementById('status').textContent = "🎉 You Win!";
      }
    } else {
      setTimeout(() => {
        firstCard.textContent = '?';
        secondCard.textContent = '?';
        firstCard.classList.remove('flipped');
        secondCard.classList.remove('flipped');
        firstCard = null;
        secondCard = null;
        lock = false;
      }, 1000);
    }
  }
}
