#!/usr/bin/env bash
set -euo pipefail

# ===== НАСТРОЙКА =====
BRANCH="main"          # дефолтная ветка твоего репо
WEEKDAYS_ONLY=false    # true = только Пн–Пт, false = все дни
MIN_COMMITS=2          # минимум коммитов в день
MAX_COMMITS=5          # максимум коммитов в день
TZ_OFFSET="+0500"      # Asia/Almaty
START_DATE="2023-07-05"
END_DATE="2023-12-29"
LOG_FILE=".activity/log.txt"
# =====================

# --- проверки ---
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "Не внутри git-репозитория. Зайди в корень проекта."; exit 1; }

git checkout "$BRANCH"

# завести «тихий» файл
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"
git add "$LOG_FILE"
git commit -m "chore: add activity log file" || true  # если уже коммитили — не падаем

# список дат через python (кроссплатформенно)
python3 - "$START_DATE" "$END_DATE" "$WEEKDAYS_ONLY" > .dates.tmp <<'PY'
import sys
from datetime import date, timedelta
start = date.fromisoformat(sys.argv[1])
end   = date.fromisoformat(sys.argv[2])
weekdays_only = sys.argv[3].lower() == "true"
d = start
while d <= end:
    if (not weekdays_only) or (d.weekday() < 5):  # 0..4 = Пн..Пт
        print(d.isoformat())
    d += timedelta(days=1)
PY

# коммиты
while read -r d; do
  n=$(( RANDOM % (MAX_COMMITS - MIN_COMMITS + 1) + MIN_COMMITS ))
  for ((i=1; i<=n; i++)); do
    echo "${d} #${i}" >> "$LOG_FILE"
    git add "$LOG_FILE"
    GIT_AUTHOR_DATE="${d} 12:00:00 ${TZ_OFFSET}" \
    GIT_COMMITTER_DATE="${d} 12:00:00 ${TZ_OFFSET}" \
    git commit -m "chore: ${d} (${i}/${n})"
  done
done < .dates.tmp
rm -f .dates.tmp

git push origin "$BRANCH"
echo "Готово: позеленены даты с ${START_DATE} по ${END_DATE} (ветка ${BRANCH})."
