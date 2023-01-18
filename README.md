# Predict Future Population
Using machine learning which is trained on population records from 1950-2023 \
The program can predict the future population.
:smile: just input the `year` and it will predict the population Using Linear Regression
You can check predictor.ipynb for the source code.
## Setup & Simple Usage:
1. Clone the repo
2. Open prediction.csv and write the years you want to predict
3. Open Jupyter Notebook and open the predictor.ipynb
4. Run the code with Juypter Notebook
5. And here you go, you can check the prediction on the prediction.csv

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "73638514",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as ny\n",
    "from sklearn import linear_model\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f4bcdd3f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>year</th>\n",
       "      <th>population</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2023</td>\n",
       "      <td>8045311447</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2022</td>\n",
       "      <td>7975105156</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2021</td>\n",
       "      <td>7909295151</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2020</td>\n",
       "      <td>7840952880</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2019</td>\n",
       "      <td>7764951032</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   year  population\n",
       "0  2023  8045311447\n",
       "1  2022  7975105156\n",
       "2  2021  7909295151\n",
       "3  2020  7840952880\n",
       "4  2019  7764951032"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('population.csv')\n",
    "df.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "881cb997",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x7f38efcfbca0>"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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6x44YcIEffHFF3rvvff0f//3f1q3bp1uueWWsMyxNo2dt9vt9rvGhw4d0pw5c5ScnKyRI0dKst+1lkJzvW+44Qbt3LlTb731lrZv365x48bpmmuu0datW33nRNv19ng8ysnJkcPh0OrVq/XJJ5/oxIkTGj16tCorK32fZad5r127VlOnTtWnn36q9957T+Xl5crJyfG7lrfffrvefvttLV26VGvXrtV3332ncePG+d4P5Hd43759GjVqlIYOHapt27ZpxowZ+u1vf6tVq1aFdb5eoZi31+TJk3XttdfW+nOKi4uVk5OjjIwMbd68WQsWLNDs2bP19NNPN9ncziQU8968ebPatWunF198UV988YX++Mc/atasWXrsscd859jtetfKxAhJZtmyZb6vd+7caSSZHTt2+I5VVFSYtLQ088wzz/iODRkyxEyfPr3Oz3333XdNXFycOXz4sO/Yk08+aVwulykrKwvpHBoq2DlfeOGF5p577qnzc//zn/8YSWbjxo2+YytWrDAOh8McPHgwtJMIQrDzrq5fv35m8uTJvq/tfK2NCX7eSUlJ5oUXXvD7rDZt2vjOicbrvWrVKhMXF2eKiop85xQWFhqHw2Hee+89Y4z9533kyBEjyaxdu9YYc2r8zZs3N0uXLvWd8+WXXxpJJi8vzxgT2O/wH/7wB5OVleX3s6699lozfPjwpp5SQIKZd1X33XefOe+882ocf+KJJ0zr1q39/l2+6667TM+ePUM/iSA0dt5ev//9783QoUN9X9v9ehtjTFSvqJxJWVmZJCkhIcF3LC4uTk6nUx9//LHfuYsXL9ZZZ52l3r17a9asWTp27Jjvvby8PPXp00ft27f3HRs+fLiKi4v1xRdfNPEsGiaQOR85ckQbNmxQu3btNHDgQLVv315Dhgzx+98kLy9PqampuuCCC3zHhg0bpri4OG3YsCFMswlcQ6611+bNm7Vt2zbddNNNvmORdK2lwOc9cOBAvfLKKzp69KgqKyu1ZMkS/fzzz7rkkkskRef1Lisrk8Ph8OszkZCQoLi4ON85dp93UVGRJKlNmzaSTv3OlpeXa9iwYb5zzjnnHHXu3Fl5eXmSAvsdzsvL8/sM7znez7BaMPMORF5eni6++GK1aNHCd2z48OHauXOnfvrppxCNPnihmndRUZHvMyT7X28pyrd+zsR7QWfNmqWffvpJJ06c0Pz58/Xtt9/q0KFDvvN+85vf6MUXX9SaNWs0a9Ys/etf/9J1113ne//w4cN+/9JL8n19+PDh8EwmQIHM+euvv5YkzZ49WzfffLNWrlypAQMG6NJLL/Xt8R8+fFjt2rXz++xmzZqpTZs2tpuzFPi1rurZZ59Vr169NHDgQN+xSLrWUuDzfvXVV1VeXq62bdvK6XTq1ltv1bJly9StWzdJ0Xm9f/WrXykpKUl33XWXjh07Jo/HozvuuEMVFRW+c+w878rKSs2YMUODBg1S7969JZ0ab4sWLZSamup3bvv27X3jDeR3uK5ziouL/erUrBDsvANh53+/QzXv9evX65VXXvHbvrTz9faK2aDSvHlzvfHGG9q1a5fatGmjxMRErVmzRiNHjvSrxbjllls0fPhw9enTRxMmTNALL7ygZcuWae/evRaOPjiBzNm7P3/rrbdq0qRJ6t+/vx566CH17NlTzz33nJXDD1qg19rr+PHjeumll/xWUyJRoPO+9957VVhYqPfff1+bNm3SzJkzdc0112j79u0Wjj54gcw7LS1NS5cu1dtvv63k5GSlpKSosLBQAwYMqPdJrnYwdepU7dixQ0uWLLF6KGHFvIOf944dO3TllVfqvvvuU05OTghH1/Qi+unJjXX++edr27ZtKioq0okTJ5SWlqYLL7zQb6m3ugsvvFCStGfPHmVmZqpDhw767LPP/M7xVl136NCh6QYfpPrm3LFjR0nSueee6/d9vXr1Un5+vqRT8zpy5Ijf+ydPntTRo0dtOWepYdf6tdde07Fjx3TDDTf4HY+0ay3VP++9e/fqscce044dO5SVlSVJOu+88/TRRx/p8ccf1z//+c+ovd45OTnau3evfvjhBzVr1kypqanq0KGDzj77bEn2/T2/7bbbfIW9VZ8e36FDB504cUKFhYV+/y+7oKDAN95Afoc7dOhQ446ZgoICuVwutWzZsimmFJDGzDsQdc3b+55VQjHv//znP7r00kt1yy236J577vF7z67Xuyr7/1+HMEhJSVFaWpp2796tTZs26corr6zz3G3btkk6/Qc9Oztb27dv9/sP2nvvvSeXy1Xjj72d1DXnLl26KD09vcZtcLt27VJGRoakU3MuLCzU5s2bfe+vXr1alZWVviBnV4Fc62effVZjxoxRWlqa3/FIvdZS3fP21ltVX0WIj4/3ra5F+/U+66yzlJqaqtWrV+vIkSMaM2aMJPvN2xij2267TcuWLdPq1avVtWtXv/fPP/98NW/eXB988IHv2M6dO5Wfn6/s7GxJgf0OZ2dn+32G9xzvZ4RbKOYdiOzsbK1bt07l5eW+Y++995569uyp1q1bN34iDRSqeX/xxRcaOnSoJk6cqLlz59b4OXa73rWyuJi3SZWUlJitW7earVu3GknmwQcfNFu3bjXffPONMcaYV1991axZs8bs3bvXLF++3GRkZJhx48b5vn/Pnj3mz3/+s9m0aZPZt2+fefPNN83ZZ59tLr74Yt85J0+eNL179zY5OTlm27ZtZuXKlSYtLc3MmjUr7PM1pvFzNsaYhx56yLhcLrN06VKze/duc88995iEhASzZ88e3zkjRoww/fv3Nxs2bDAff/yx6d69uxk/fnxY51pVKOZtjDG7d+82DofDrFixosZ7drvWxjR+3idOnDDdunUzgwcPNhs2bDB79uwxf//7343D4TDvvPOO77xovN7PPfecycvLM3v27DH/+te/TJs2bczMmTP9zrHTvH/3u9+ZlJQU8+GHH5pDhw75XseOHfOdM2XKFNO5c2ezevVqs2nTJpOdnW2ys7N97wfyO/z111+bxMREc+edd5ovv/zSPP744yY+Pt6sXLkyrPP1CsW8jTn17/bWrVvNrbfeanr06OH7/fHe5VNYWGjat29vrr/+erNjxw6zZMkSk5iYaJ566qmwztcrFPPevn27SUtLM9ddd53fZxw5csR3jt2ud22iOqisWbPGSKrxmjhxojHGmEceecR06tTJNG/e3HTu3Nncc889frem5efnm4svvti0adPGOJ1O061bN3PnnXf63dJojDH79+83I0eONC1btjRnnXWW+d///V9TXl4ezqn6NHbOXrm5uaZTp04mMTHRZGdnm48++sjv/R9//NGMHz/eJCcnG5fLZSZNmmRKSkrCMcVahWres2bNMm6321RUVNT6c+x0rY0Jzbx37dplxo0bZ9q1a2cSExNN3759a9yuHI3X+6677jLt27c3zZs3N927dzcPPPCAqays9DvHTvOubb6SzMKFC33nHD9+3Pz+9783rVu3NomJieaqq64yhw4d8vucQH6H16xZY/r162datGhhzj77bL+fEW6hmveQIUNq/Zx9+/b5zvn888/NRRddZJxOp/nFL35h5s2bF6ZZ1hSKed933321fkZGRobfz7LT9a4NT08GAAC2RY0KAACwLYIKAACwLYIKAACwLYIKAACwLYIKAACwLYIKAACwLYIKAACwLYIKAACwLYIKAACwLYIKgCZljNGwYcM0fPjwGu898cQTSk1N1bfffmvByABEAoIKgCblcDi0cOFCbdiwQU899ZTv+L59+/SHP/xBjz76qN/j60Oh6hNwAUQ2ggqAJud2u/XII4/ojjvu0L59+2SM0U033aScnBz1799fI0eOVHJystq3b6/rr79eP/zwg+97V65cqYsuukipqalq27atrrjiCu3du9f3/v79++VwOPTKK69oyJAhSkhI0OLFi62YJoAmwEMJAYTN2LFjVVRUpHHjxun+++/XF198oaysLP32t7/VDTfcoOPHj+uuu+7SyZMntXr1aknS66+/LofDob59+6q0tFR/+tOftH//fm3btk1xcXHav3+/unbtqi5duuiBBx5Q//79lZCQoI4dO1o8WwChQFABEDZHjhxRVlaWjh49qtdff107duzQRx99pFWrVvnO+fbbb+V2u7Vz50716NGjxmf88MMPSktL0/bt29W7d29fUHn44Yc1ffr0cE4HQBiw9QMgbNq1a6dbb71VvXr10tixY/X5559rzZo1Sk5O9r3OOeccSfJt7+zevVvjx4/X2WefLZfLpS5dukiS8vPz/T77ggsuCOtcAIRHM6sHACC2NGvWTM2anfpPT2lpqUaPHq358+fXOM+7dTN69GhlZGTomWeeUXp6uiorK9W7d2+dOHHC7/ykpKSmHzyAsCOoALDMgAED9Prrr6tLly6+8FLVjz/+qJ07d+qZZ57R4MGDJUkff/xxuIcJwEJs/QCwzNSpU3X06FGNHz9eGzdu1N69e7Vq1SpNmjRJFRUVat26tdq2baunn35ae/bs0erVqzVz5kyrhw0gjAgqACyTnp6uTz75RBUVFcrJyVGfPn00Y8YMpaamKi4uTnFxcVqyZIk2b96s3r176/bbb9eCBQusHjaAMOKuHwAAYFusqAAAANsiqAAAANsiqAAAANsiqAAAANsiqAAAANsiqAAAANsiqAAAANsiqAAAANsiqAAAANsiqAAAANsiqAAAANsiqAAAANv6fzfHqNDrdTxrAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "%matplotlib inline\n",
    "plt.xlabel('Year')\n",
    "plt.ylabel('Population')\n",
    "plt.scatter(df['year'],df['population'],marker=\"+\",color=\"red\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ed7e7593",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "req = linear_model.LinearRegression()\n",
    "req.fit(df[['year']],df.population)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f909504e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>year</th>\n",
       "      <th>population</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>10001</td>\n",
       "      <td>639462579209</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3</td>\n",
       "      <td>5000</td>\n",
       "      <td>243617652283</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4</td>\n",
       "      <td>4000</td>\n",
       "      <td>164464497528</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>5</td>\n",
       "      <td>3000</td>\n",
       "      <td>85311342774</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6</td>\n",
       "      <td>2300</td>\n",
       "      <td>29904134446</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0   year    population\n",
       "0           0  10001  639462579209\n",
       "1           3   5000  243617652283\n",
       "2           4   4000  164464497528\n",
       "3           5   3000   85311342774\n",
       "4           6   2300   29904134446"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pr = pd.read_csv('prediction.csv')\n",
    "pr.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "f799a31b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>year</th>\n",
       "      <th>population</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10001</td>\n",
       "      <td>6.394626e+11</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    year    population\n",
       "0  10001  6.394626e+11"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pr['population'] = req.predict(pr[['year']])\n",
    "pr.head(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "9ff74b18",
   "metadata": {},
   "outputs": [],
   "source": [
    "pr.to_csv(\"prediction.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0be9acfe",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
