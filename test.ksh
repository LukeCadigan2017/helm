#model, num_beams, num_eval
# ./test_run_all.ksh distilbert/distilgpt2 2 1


DISABLE_CACHE="${DISABLE_CACHE:=true}"
echo "DISABLE_CACHE IS $DISABLE_CACHE"

if [ "$DISABLE_CACHE" = true ] ; then
   echo "Disable cache"
else
   echo "Do not disable cache"
fi