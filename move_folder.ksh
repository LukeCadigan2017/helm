

export NEW_CACHE_BASE=/scratch-local/lcadigan/cache
export NEW_CACHE=${NEW_CACHE_BASE}/huggingface/
export PREV_CACHE=~/.cache/huggingface

echo ls $PREV_CACHE
ls $PREV_CACHE

echo ls $NEW_CACHE_BASE
ls $NEW_CACHE_BASE

echo mkdir -p $NEW_CACHE_BASE
mkdir -p $NEW_CACHE_BASE


echo mv $PREV_CACHE $NEW_CACHE



mv $PREV_CACHE $NEW_CACHE




#add to setup_env.ksh
echo export HF_HOME=$NEW_CACHE
export HF_HOME=$NEW_CACHE