/*
SQLyog Community Edition- MySQL GUI v7.01 
MySQL - 5.0.27-community-nt : Database - resumeverificationtwo
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

CREATE DATABASE /*!32312 IF NOT EXISTS*/`resumeverificationtwo` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `resumeverificationtwo`;

/*Table structure for table `answers` */

DROP TABLE IF EXISTS `answers`;

CREATE TABLE `answers` (
  `id` int(11) NOT NULL auto_increment,
  `sid` varchar(30) default NULL,
  `answers` varchar(7000) default NULL,
  UNIQUE KEY `id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `answers` */

insert  into `answers`(`id`,`sid`,`answers`) values (191,'4','this is Prathamesh from inbotics'),(192,'4','this is Prathamesh from in politics'),(193,'4','Prathamesh from in vertex'),(194,'4','this is Prathamesh from inbotics'),(195,'4','Prathamesh from inbotics');

/*Table structure for table `marks` */

DROP TABLE IF EXISTS `marks`;

CREATE TABLE `marks` (
  `id` int(11) NOT NULL auto_increment,
  `sid` varchar(11) default NULL,
  `studname` varchar(11) default NULL,
  `q1` varchar(11) default NULL,
  `q2` varchar(11) default NULL,
  `q3` varchar(11) default NULL,
  `q4` varchar(11) default NULL,
  `q5` varchar(11) default NULL,
  `marks` varchar(11) default NULL,
  UNIQUE KEY `id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `marks` */

insert  into `marks`(`id`,`sid`,`studname`,`q1`,`q2`,`q3`,`q4`,`q5`,`marks`) values (4,'prathamesh',NULL,'0.6','0.5','0.7','1.1','0.6','4');

/*Table structure for table `student` */

DROP TABLE IF EXISTS `student`;

CREATE TABLE `student` (
  `id` int(11) NOT NULL auto_increment,
  `name` varchar(40) default NULL,
  `mobile` int(11) default NULL,
  `email` varchar(40) default NULL,
  `pass` varchar(40) default NULL,
  UNIQUE KEY `id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `student` */

insert  into `student`(`id`,`name`,`mobile`,`email`,`pass`) values (1,'Prakash',0,'0000011111','123'),(2,'man',0,'7836814663','123'),(3,'manish',2147483647,'manish@gmail.com','123'),(4,'sushant',2147483647,'sushant@gmail.com','123'),(5,'chetan',2147483647,'chetan@gmail.com','123');

/*Table structure for table `teacher` */

DROP TABLE IF EXISTS `teacher`;

CREATE TABLE `teacher` (
  `id` int(11) NOT NULL auto_increment,
  `email` varchar(40) default NULL,
  `password` varchar(40) default NULL,
  UNIQUE KEY `id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `teacher` */

insert  into `teacher`(`id`,`email`,`password`) values (1,'prathamesh@gmail.com','123');

/*Table structure for table `teacherquestion` */

DROP TABLE IF EXISTS `teacherquestion`;

CREATE TABLE `teacherquestion` (
  `qid` int(11) NOT NULL auto_increment,
  `question` varchar(1500) default NULL,
  `answer` varchar(6000) default NULL,
  `marks` varchar(11) default NULL,
  UNIQUE KEY `qid` (`qid`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `teacherquestion` */

insert  into `teacherquestion`(`qid`,`question`,`answer`,`marks`) values (1,'when a project or assignment didn’t go as planned. How would you approach the situation differently in the future?','Last semester we had a group project that took approximately six weeks. Around week four, we realized that one of the group members was not pulling his weight. The work he agreed to do was not getting done. I took charge of the situation and scheduled a group meeting to discuss the issue. Ultimately, that person dropped the course, but by addressing the problem head-on, the group was able to divide up our work and complete the project on-time. In the future, I would make sure that the group has weekly meetings to assess our progress. That would make sure the project was on-track and that the work was getting done.','10'),(2,'What do you enjoy most and least about engineering?','I really love the design work in engineering, the face-to-face interaction with clients, and the opportunity to see projects come to life. But if I had to pick one thing that I don’t enjoy as much, I would have to say it’s contract preparation','10'),(3,'Where do you see yourself five years from now?','Your new employer is going to invest a lot of time and money in your training and development, and they don’t want to hear that you get bored easily and will likely look for opportunities elsewhere before too long','10'),(4,'What new engineering skills have you recently developed?','Since graduating, I’ve been searching for work and also training myself on Civil 3D. I have a basic knowledge of Civil 3D from school, but I thought upgrading my skills would be a valuable investment in my career as an engineer','10'),(5,'Why are you interested in a position with our company?','Your interviewer is trying to understand “Why us?” This is your chance to tell him what you know about the company and express a genuine enthusiasm for the role. Take a look at the company website and any recent press releases.','10');

/*Table structure for table `user_company_information` */

DROP TABLE IF EXISTS `user_company_information`;

CREATE TABLE `user_company_information` (
  `id` int(11) NOT NULL auto_increment,
  `username` varchar(100) default NULL,
  `company_name` longtext,
  `start_date` varchar(100) default NULL,
  `End_date` varchar(100) default NULL,
  `technology_worked` longtext,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `user_company_information` */

insert  into `user_company_information`(`id`,`username`,`company_name`,`start_date`,`End_date`,`technology_worked`) values (1,'a','Resumeverifier','f','f','f'),(2,'ningesh','Resumeverifier','f','f','f'),(3,'a','Resumeverifier','f','f','f'),(4,'a','Resumeverifier','f','f','f'),(5,'b','Resumeverifier','f','f','f'),(6,'a','Resumeverifier','f','f','f'),(7,'ab','Resumeverifier','f','f','f'),(8,'ab','Resumeverifier','f','f','f'),(9,'a','Resumeverifier','f','f','f'),(10,'a','Resumeverifier','f','f','f'),(11,'ab','Resumeverifier','f','f','f'),(12,'a','Resumeverifier','f','f','f'),(13,'ab','Resumeverifier','f','f','f'),(14,'a','Resumeverifier','f','f','f'),(15,'ab','Resumeverifier','f','f','f'),(16,'a','Resumeverifier','f','f','f'),(17,'a','Resumeverifier','f','f','f'),(18,'a','Resumeverifier','f','f','f'),(19,'ab','Resumeverifier','f','f','f'),(20,'a','Resumeverifier','f','f','f'),(21,'ab','Resumeverifier','f','f','f'),(22,'a','Resumeverifier','f','f','f'),(23,'a','Resumeverifier','f','f','f'),(24,'ab','Resumeverifier','f','f','f'),(25,'ningesh','Resumeverifier','f','f','f'),(26,'admin','Resumeverifier','f','f','f'),(27,'admin','k','k','k','k'),(28,'Prathamesh','Projectwale','f','f','Java'),(29,'yash','Projectwale','02/02/2021','25/04/2021','Java');

/*Table structure for table `userdetails` */

DROP TABLE IF EXISTS `userdetails`;

CREATE TABLE `userdetails` (
  `id` int(11) NOT NULL auto_increment,
  `name` varchar(100) default NULL,
  `address` varchar(500) default NULL,
  `mobileno` varchar(20) default NULL,
  `emailid` varchar(100) default NULL,
  `password` varchar(100) default NULL,
  UNIQUE KEY `id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `userdetails` */

insert  into `userdetails`(`id`,`name`,`address`,`mobileno`,`emailid`,`password`) values (1,'Sujay','navi mumbai','7894561230','sujay@gmail.com','123'),(2,'Abc','navi mumbai','7894561230','abc@gmail.com','123'),(3,'Sujay','navi mumbai','7894561230','sujay@gmail','123'),(4,'prathamesh','india','9090909090','prathameshmane852@gmail.com','12345');

/*Table structure for table `videoemotions` */

DROP TABLE IF EXISTS `videoemotions`;

CREATE TABLE `videoemotions` (
  `id` int(11) NOT NULL auto_increment,
  `name` varchar(11) default NULL,
  `Angry` varchar(11) default NULL,
  `Disgusted` varchar(11) default NULL,
  `Fearful` varchar(11) default NULL,
  `Happy` varchar(11) default NULL,
  `Neutral` varchar(11) default NULL,
  `sad` varchar(11) default NULL,
  `surprise` varchar(11) default NULL,
  UNIQUE KEY `id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `videoemotions` */

insert  into `videoemotions`(`id`,`name`,`Angry`,`Disgusted`,`Fearful`,`Happy`,`Neutral`,`sad`,`surprise`) values (23,'prathamesh','0','0','6','32','24','0','188');

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
